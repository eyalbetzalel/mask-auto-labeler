# Domain Adaptation Pre Training

# Copyright (c) 2021-2022, NVIDIA Corporation & Affiliates. All rights reserved. 
# 
# This work is made available under the Nvidia Source Code License-NC. 
# To view a copy of this license, visit 
# https://github.com/NVlabs/MAL/blob/main/LICENSE


import time
import itertools
import json
import os
import math

import cv2
import numpy as np

import torch
from torch import nn, optim

from PIL import Image

from pycocotools.coco import COCO
from pycocotools.mask import encode
from pycocotools.cocoeval import COCOeval

from mmcv.cnn import ConvModule

import torchmetrics
import pytorch_lightning as pl

from fairscale.nn import checkpoint_wrapper, auto_wrap, wrap

import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions.beta import Beta as BetaDist
from torch.distributions.kl import kl_divergence
from torch.distributed import all_reduce, ReduceOp
import torch.distributed as dist
from torchvision import transforms

from . import vision_transformers
from datasets.data_aug import Denormalize
from datasets.pl_data_module import datapath_configs, num_class_dict
from utils.optimizers.adamw import AdamWwStep

from models.MultiModelCrf import visualize_and_save_feature_map, visualize_and_save_depth_map, visualize_and_save_batch, visualize_and_save_all
from models.prismer.experts.generate_depth import model as model_depth

from torchvision.utils import make_grid
import random
import wandb
import matplotlib.pyplot as plt

model_depth0 = model_depth.cuda(0)
# model_depth1 = model_depth.cuda(1)
iou_arr = []


class MaskHead(nn.Module):

    def __init__(self, in_channels=2048, args=None):
        super().__init__()
        self.num_convs                  = args.mask_head_num_convs
        self.in_channels                = in_channels
        self.mask_head_hidden_channel   = args.mask_head_hidden_channel
        self.mask_head_out_channel      = args.mask_head_out_channel
        self.mask_scale_ratio           = args.mask_scale_ratio

        self.convs = nn.ModuleList()
        for i in range(self.num_convs):
            in_channels = self.in_channels if i == 0 else self.mask_head_hidden_channel
            out_channels = self.mask_head_hidden_channel if i < self.num_convs - 1 else self.mask_head_out_channel
            self.convs.append(ConvModule(in_channels, out_channels, 3, padding=1))

    def forward(self, x):
        for idx, conv in enumerate(self.convs):
            if idx == 3:
                h, w = x.shape[2:]
                th, tw = int(h * self.mask_scale_ratio), int(w * self.mask_scale_ratio)
                x = F.interpolate(x, (th, tw), mode='bilinear', align_corners=False)
            x = conv(x)
        return x

class RoIHead(nn.Module):

    def __init__(self, in_channels=2048, args=None):
        super().__init__()
        self.mlp1 = nn.Linear(in_channels, args.mask_head_out_channel)
        self.relu = nn.ReLU()
        self.mlp2 = nn.Linear(args.mask_head_out_channel, args.mask_head_out_channel)
    
    def forward(self, x, boxmask=None):
        x = x.mean((2, 3))
        x = self.mlp2(self.relu(self.mlp1(x)))
        return x

class MALStudentNetwork(pl.LightningModule):

    def __init__(self, in_channels=2048, args=None):
        super().__init__()
        self.args = args
        self.backbone               = vision_transformers.get_vit(args=args)
        mask_head_num_convs         = args.mask_head_num_convs
        mask_head_hidden_channel    = args.mask_head_hidden_channel
        mask_head_out_channel       = args.mask_head_out_channel

        # K head
        self.roi_head = RoIHead(in_channels, args=args)
        # V head
        self.mask_head = MaskHead(in_channels, args=args)

        # make student sharded on multiple gpus
        self.configure_sharded_model()

    def configure_sharded_model(self):
        self.backbone = auto_wrap(self.backbone)
    
    def forward(self, x, boxmask, bboxes):
        x = x.half()
        feat = self.backbone.base_forward(x)
        spatial_feat_ori = self.backbone.get_spatial_feat(feat)
        h, w = spatial_feat_ori.shape[2:]
        mask_scale_ratio_pre = int(self.args.mask_scale_ratio_pre)
        if not self.args.not_adjust_scale:
            spatial_feat_list = []
            masking_list = []
            areas = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
            for idx, (scale_low, scale_high) in enumerate([(0, 32**2), (32**2, 96**2), (96**2, 1e5**2)]):
                masking = (areas < scale_high) * (areas > scale_low)
                if masking.sum() > 0:
                    spatial_feat = F.interpolate(spatial_feat_ori[masking], 
                                            size=(int(h*2**(idx-1)), int(w*2**(idx-1))),
                                            mode='bilinear', align_corners=False)
                    boxmask = None
                else:
                    spatial_feat = None
                    boxmask = None
                spatial_feat_list.append(spatial_feat)
                masking_list.append(masking)
            roi_feat = self.roi_head(spatial_feat_ori)
            n, maxh, maxw = roi_feat.shape[0], h * 4, w * 4 
            seg_list = []
            seg_all = torch.zeros(n, 1, maxh, maxw).to(roi_feat)
            for idx, (spatial_feat, masking) in enumerate(zip(spatial_feat_list, masking_list)):
                if masking.sum() > 0:
                    mn = masking.sum()
                    mh, mw = int(h * mask_scale_ratio_pre * 2**(idx-1)), int(w * mask_scale_ratio_pre * 2**(idx-1))
                    seg_feat = self.mask_head(spatial_feat)
                    c = seg_feat.shape[1]
                    masked_roi_feat = roi_feat[masking]
                    seg = (masked_roi_feat[:, None, :] @ seg_feat.reshape(mn, c, mh * mw * 4)).reshape(mn, 1, mh * 2, mw * 2)
                    seg = F.interpolate(seg, size=(maxh, maxw), mode='bilinear', align_corners=False)
                    seg_all[masking] = seg
            
            ret_vals = {'feat': feat, 'seg': seg_all, 'spatial_feat': spatial_feat_ori, 'masking_list': masking_list}
        else:
            spatial_feat    = F.interpolate(spatial_feat_ori, size=(int(h*self.args.mask_scale_ratio_pre), int(w*self.args.mask_scale_ratio_pre)),
                                        mode='bilinear', align_corners=False)
            boxmask         = F.interpolate(boxmask, size=spatial_feat.shape[2:], mode='bilinear', align_corners=False)
            seg_feat        = self.mask_head(spatial_feat)
            roi_feat        = self.roi_head(spatial_feat_ori, boxmask)
            n, c, h, w      = seg_feat.shape
            seg             = (roi_feat[:,None,:] @ seg_feat.reshape(n, c, h * w)).reshape(n, 1, h, w)
            seg             = F.interpolate(seg, (h * 4, w * 4), mode='bilinear', align_corners=False)
            ret_vals = {'feat': feat, 'seg': seg, 'spatial_feat': spatial_feat_ori}
        return ret_vals   

class MIoUMetrics(torchmetrics.Metric):

    def __init__(self, dist_sync_on_step=True, num_classes=20):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("cnt", default=torch.zeros(num_classes), dist_reduce_fx="sum")
        self.add_state("total", default=torch.zeros(num_classes), dist_reduce_fx="sum")

    def update(self, label, iou):
        self.cnt[label-1] += 1
        self.total[label-1] += iou

    def update_with_ious(self, labels, ious):
        for iou, label in zip(ious, labels):
            self.cnt[label-1] += 1
            self.total[label-1] += float(iou)
        return ious
    

    def cal_intersection(self, seg, gt):
        B = seg.shape[0]
        inter_cnt = (seg * gt).reshape(B, -1).sum(1)
        return inter_cnt
    
    def cal_union(self, seg, gt, inter_cnt=None):
        B = seg.shape[0]
        if inter_cnt is None:
            inter_cnt = self.cal_intersection(seg, gt)
        union_cnt = seg.reshape(B, -1).sum(1) + gt.reshape(B, -1).sum(1) - inter_cnt
        return union_cnt
    
    def cal_iou(self, seg, gt):
        inter_cnt = self.cal_intersection(seg, gt)
        union_cnt = self.cal_union(seg, gt, inter_cnt)
        return 1.0 * inter_cnt / (union_cnt + 1e-6)

    def compute(self):
        mIoUs = self.total / (1e-6 + self.cnt)
        mIoU = mIoUs.sum() / (self.cnt > 0).sum()
        return mIoU

    def compute_with_ids(self, ids=None):
        if ids is not None:
            total = self.total[torch.tensor(np.array(ids)).long()]
            cnt = self.cnt[torch.tensor(np.array(ids)).long()]
        else:
            total = self.total
            cnt = self.cnt
        mIoUs = total / (1e-6 + cnt)
        mIoU = mIoUs.sum() / (cnt > 0).sum()
        return mIoU

class DAPT(pl.LightningModule):
    
    def __init__(self, args=None, num_iter_per_epoch=None):

        super().__init__()

        # mIoU torchmetrics

        # loss term hyper parameters
        self.num_convs = args.mask_head_num_convs
        self.args = args

        self.mask_thres = args.mask_thres

        self.num_classes = num_class_dict[args.dataset_type]

        self.mIoUMetric = MIoUMetrics(num_classes=self.num_classes)
        self.areaMIoUMetrics = nn.ModuleList([MIoUMetrics(num_classes=self.num_classes) for _ in range(3)])
        if self.args.comp_clustering:
            self.clusteringScoreMetrics = torchmetrics.MeanMetric()

        backbone_type = args.arch


        if 'tiny' in backbone_type.lower():
            in_channel = 192
        if 'small' in backbone_type.lower():
            in_channel = 384
        elif 'base' in backbone_type.lower():
            in_channel = 768
        elif 'large' in backbone_type.lower():
            in_channel = 1024
        elif 'huge' in backbone_type.lower():
            in_channel = 1280


        self.student = MALStudentNetwork(in_channel, args=args)

        self.denormalize = Denormalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

        # optimizer parameters
        self._optim_type = args.optim_type
        self._lr = args.lr
        self._wd = args.wd
        self._momentum = args.optim_momentum
        if num_iter_per_epoch is not None:
            self._num_iter_per_epoch = num_iter_per_epoch // len(self.args.gpus)

        self.args = args

        self.vis_cnt = 0
        self.local_step = 0

        # Enable manual optimization
        self.automatic_optimization = False

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self._lr, momentum=0.9)
        return optimizer 

    def dice_loss(self, input, target):
        input = input.contiguous().view(input.size()[0], -1).float()
        target = target.contiguous().view(target.size()[0], -1).float()

        a = torch.sum(input * target, 1)
        b = torch.sum(input * input, 1) + 0.001
        c = torch.sum(target * target, 1) + 0.001
        d = (2 * a) / (b + c)

        return 1-d

    def training_step(self, x):
        optimizer = self.optimizers()
        loss = {}
        image = x['image']

        local_step = self.local_step
        self.local_step += 1

        if 'timage' in x.keys():
            timage = x['timage']
        else:
            timage = image
        student_output = self.student(image, x['mask'], x['bbox'])
        # teacher_output = self.teacher(timage, x['mask'], x['bbox'])
        B, oh, ow = student_output['seg'].shape[0], student_output['seg'].shape[2], student_output['seg'].shape[3]
        gt_mask  = F.interpolate(x['gt_mask'], size=(oh, ow), mode='bilinear', align_corners=False).reshape(-1, oh, ow)
        
        args = self.args


        if 'image' in x:
            student_seg_sigmoid = torch.sigmoid(student_output['seg'])[:,0].float()
            #teacher_seg_sigmoid = torch.sigmoid(teacher_output['seg'])[:,0].float()
            loss_dice = self.dice_loss(student_seg_sigmoid, gt_mask)
            loss_dice = loss_dice.sum() / (loss_dice.numel() + 1e-4)
            loss.update({'dice': loss_dice})
            #self.log("train/loss_dice", loss_dice, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
        else:
            raise NotImplementedError

        total_loss = sum(loss.values())
            # Log using wandb
        wandb.log({
            "train/loss": total_loss,
            "lr": optimizer.param_groups[0]['lr'],
            "train/bs": image.shape[0],
            "epoch" : self.current_epoch
        })
        optimizer.zero_grad()
        self.manual_backward(total_loss)
        for name, param in self.student.named_parameters():
            assert not torch.isnan(param.grad).any()
        optimizer.step()
        if self._optim_type == 'adamw':
            self.set_lr_per_iteration(optimizer, 1. * local_step)

        self.last_output = {
            "image" : image,
            "bbox" : x['bbox'],
            "gt_mask" : x['gt_mask'],
            "seg" : (student_seg_sigmoid >= 0.5).float()
        }
    
    def set_lr_per_iteration(self, optimizer, local_step):
        args = self.args
        epoch = 1. * local_step / self._num_iter_per_epoch + self.current_epoch
        if epoch < args.warmup_epochs:
            lr = args.lr * (epoch / args.warmup_epochs)
        else:
            lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
                (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.max_epochs - args.warmup_epochs) * args.num_wave))
        for param_group in optimizer.param_groups:
            if "lr_scale" in param_group:
                param_group["lr"] = lr * param_group["lr_scale"]
            else:
                param_group["lr"] = lr
        return lr

    def training_epoch_end(self, outputs):
        optimizer = self.optimizers()
        self.local_step = 0

        # Sample 5 random images and their corresponding segmentation maps for visualization
        images, segmentations, sampled_gt_seg = self.sample_images_and_segmentations()
        examples = []
        
        for idx, (image, segmentation, gt_seg) in enumerate(zip(images, segmentations, sampled_gt_seg)):
            overlay_image = self.overlay_segmentation_on_image_2(image, gt_seg, segmentation)
            examples.append(wandb.Image(overlay_image, caption=f"Ep. {self.current_epoch + 1} | ID {idx + 1}"))
        wandb.log({"Images": examples})
        
        # Save model weights every args.save_dapt_cp epochs
        if (self.current_epoch + 1) % self.args.save_dapt_cp_every_x_epochs == 0:
            checkpoint_name = self.args.dapt_cp_path
            self.trainer.save_checkpoint(checkpoint_name)



    
    def sample_images_and_segmentations(self, num_samples=5):
        # This method should return pairs of original images and their segmentation maps 
        # For simplicity, let's assume `self.last_output` is a dictionary 
        # that contains 'image' and 'seg' keys which store batch of images and segmentations

        indices = random.sample(range(self.last_output['image'].shape[0]), num_samples)

        sampled_images = [self.last_output['image'][i] for i in indices]
        sampled_segmentations = [self.last_output['seg'][i] for i in indices]
        sampled_gt_seg = [self.last_output['gt_mask'][i] for i in indices] 
        return sampled_images, sampled_segmentations, sampled_gt_seg
    
    def overlay_segmentation_on_image(self, image, segmentation):
        """
        Overlay the segmentation on top of the image.
        Assumes both image and segmentation are torch tensors.
        """
        image = (image - image.min()) / (image.max() - image.min())

        C, oh, ow = image.shape
        segmentation = segmentation[None,None,...]
        segmentation  = F.interpolate(segmentation, size=(oh, ow), mode='bilinear', align_corners=False).reshape(-1, oh, ow)
        
        # Convert tensors to numpy
        image_np = image.permute(1, 2, 0).cpu().numpy()
        segmentation_np = segmentation.squeeze(0).cpu().numpy()

        # Blend the segmentation mask onto the image
        alpha = 0.3  # Adjust the blending strength
        overlayed_image = (1 - alpha) * image_np + alpha * segmentation_np[..., None]

        return overlayed_image

    def overlay_segmentation_on_image_2(self, image, ground_truth_segmentation, predicted_segmentation):
        """
        Overlay the ground truth and predicted segmentation on top of the image.
        Assumes image, ground_truth_segmentation, and predicted_segmentation are torch tensors.
        """
        image = (image - image.min()) / (image.max() - image.min())

        C, oh, ow = image.shape
        
        ground_truth_segmentation = ground_truth_segmentation[None,...]
        #ground_truth_segmentation  = F.interpolate(ground_truth_segmentation, size=(oh, ow), mode='bilinear', align_corners=False).squeeze(0)
        
        predicted_segmentation = predicted_segmentation[None,None,...]
        predicted_segmentation = F.interpolate(predicted_segmentation, size=(oh, ow), mode='bilinear', align_corners=False).squeeze(0)
        
        # Convert tensors to numpy
        image_np = image.permute(1, 2, 0).cpu().numpy()
        gt_np = ground_truth_segmentation.cpu().numpy()
        pred_np = predicted_segmentation.cpu().numpy()

        # Create the overlayed images
        gt_colored = np.stack([np.zeros_like(gt_np), gt_np, np.zeros_like(gt_np)], axis=-1)
        pred_colored = np.stack([pred_np, np.zeros_like(pred_np), np.zeros_like(pred_np)], axis=-1)

        gt_colored = np.squeeze(gt_colored)
        pred_colored = np.squeeze(pred_colored)

        alpha = 0.3  # Adjust the blending strength
        overlayed_gt = (1 - alpha) * image_np + alpha * gt_colored
        overlayed_pred = (1 - alpha) * image_np + alpha * pred_colored
        
        # Stack images vertically
        combined_img = np.vstack([overlayed_gt, overlayed_pred])
        combined_img = (combined_img * 255).astype(np.uint8)

        # Add text labels using OpenCV
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.5
        thickness = 2
        color_text = (255, 255, 255)  # white
        cv2.putText(combined_img, "Ground Truth", (10, 25), font, scale, color_text, thickness)
        cv2.putText(combined_img, "Predicted", (10, oh + 25), font, scale, color_text, thickness)

        return combined_img