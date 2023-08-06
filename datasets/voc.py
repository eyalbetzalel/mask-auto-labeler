# Copyright (c) 2021-2022, NVIDIA Corporation & Affiliates. All rights reserved. 
# 
# This work is made available under the Nvidia Source Code License-NC. 
# To view a copy of this license, visit 
# https://github.com/NVlabs/MAL/blob/main/LICENSE


from re import T
import torch
from torch.utils.data import Dataset
import numpy as np

from pycocotools.coco import COCO
import pycocotools.mask as maskUtils

from PIL import Image

import xml.etree.ElementTree as ET
import lmdb

import os
import json
import cv2
import pickle

import time

from models.prismer.experts.generate_depth import model as model_depth
import torchvision.transforms as transforms


# # Load depth map from file: 

# depth_map_path = "/workspace/mask-auto-labeler/data/cityscapes/depth_maps.pt"
# depth_map = torch.load(depth_map_path, map_location=torch.device('cpu'))

CLASS_NAMES = [
		"aeroplane",
		"bicycle",
		"bird",
		"boat",
		"bottle",
		"bus",
		"car",
		"cat",
		"chair",
		"cow",
		"diningtable",
		"dog",
		"horse",
		"motorbike",
		"person",
		"pottedplant",
		"sheep",
		"sofa",
		"train",
		"tvmonitor"]
CLASS2IDX = dict([ (cn, idx) for idx, cn in enumerate(CLASS_NAMES)])


class DataWrapper:

    def __init__(self, data):
        self.data = data


    

class BoxLabelVOC(Dataset):

    def __init__(self, ann_path, img_data_dir, min_obj_size=0, max_obj_size=1e10, transform=None, args=None):
        self.args = args
        self.ann_path = ann_path
        self.img_data_dir = img_data_dir
        self.min_obj_size = min_obj_size
        self.max_obj_size = max_obj_size
        self.transform = transform
        self.coco = COCO(ann_path)
        self._filter_imgs()
        self.get_category_mapping()
    
    def save_mask_and_bbox(self, img_path, mask, bbox, output_path):
        # Load the image and mask
        img = self.get_image(img_path)
        img = np.array(img)[:, :, ::-1]
        mask = mask.astype(np.uint8)
        mask[mask == 1] = 255

        # Create a 3-channel version of the mask
        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        # Create a 3-channel version of the mask with color blue
        #mask_3ch = np.zeros_like(img)
        #mask_3ch[bbox[1]:bbox[3], bbox[0]:bbox[2], 0] = mask[bbox[1]:bbox[3], bbox[0]:bbox[2]]  # Blue channel

        # Overlay the mask onto the image
        combined_img = cv2.addWeighted(img, 0.7, mask_3ch, 0.3, 0)

        # Draw the bounding box on the combined image
        # Bounding box format is [xmin, ymin, xmax, ymax]
        x0, y0, x1, y1 = int(bbox[0]), int(bbox[1]), int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])
        
        cv2.rectangle(combined_img, (x0, y0), (x1, y1), (0, 255, 0), 2)

        # Save the combined image
        cv2.imwrite(output_path, combined_img)

<<<<<<< HEAD
    # def get_cs_mask(self, img_path, bbox):
    #     # # Replace parts of the path to get the path to the ground truth
    #     # mask_path = img_path.replace("leftImg8bit", "gtFine")
    #     # mask_path = os.path.splitext(mask_path)[0] + "_instanceIds.png"
    #     # mask_path = os.path.join("data/cityscapes/", mask_path)
    #     # # Load the mask for specific instance_id:
    #     # mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)  # Load with original depth
    #     # x0, y0, x1, y1 = int(bbox[0]), int(bbox[1]), int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])
    #     # bbox_mask = mask[y0:y1+1, x0:x1+1]
    #     # unique_ids, counts = np.unique(bbox_mask, return_counts=True)
    #     # max_count_id = unique_ids[np.argmax(counts)]
    #     # binary_mask = np.where(mask == max_count_id, 1, 0)
    #     # self.save_mask_and_bbox(img_path=img_path, mask=binary_mask, bbox=bbox, output_path="./test_mask.png")
    #     # return binary_mask

    #     mask_path = img_path.replace("leftImg8bit", "gtFine")
    #     mask_path = os.path.splitext(mask_path)[0] + "_instanceIds.png"
    #     mask_path = os.path.join("data/cityscapes/", mask_path)
    #     # Load the mask for specific instance_id:
    #     mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)  # Load with original depth
    #     x0, y0, x1, y1 = int(bbox[0]), int(bbox[1]), int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])
    #     bbox_mask = mask[y0:y1+1, x0:x1+1]
    #     unique_ids = np.unique(bbox_mask)
    #     min_l2_dist = float('inf')
    #     chosen_id = None
    #     for uid in unique_ids:
    #         y, x = np.where(bbox_mask == uid)
    #         if len(x) == 0 or len(y) == 0:
    #             continue
    #         instance_bbox = [np.min(x), np.min(y), np.max(x) - np.min(x), np.max(y) - np.min(y)]
    #         instance_bbox = [bbox[0] + instance_bbox[0], bbox[1] + instance_bbox[1], instance_bbox[2], instance_bbox[3]]
    #         l2_dist = np.sqrt(np.sum((np.array(instance_bbox) - np.array(bbox))**2))
    #         if l2_dist < min_l2_dist:
    #             min_l2_dist = l2_dist
    #             chosen_id = uid
    #     binary_mask = np.where(mask == chosen_id, 1, 0)
    #     self.save_mask_and_bbox(img_path=img_path, mask=binary_mask, bbox=bbox, output_path="./test_mask.png")
    #     return binary_mask

=======
    def get_cs_mask(self, img_path, bbox):
        # # Replace parts of the path to get the path to the ground truth
        # mask_path = img_path.replace("leftImg8bit", "gtFine")
        # mask_path = os.path.splitext(mask_path)[0] + "_instanceIds.png"
        # mask_path = os.path.join("data/cityscapes/", mask_path)
        # # Load the mask for specific instance_id:
        # mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)  # Load with original depth
        # x0, y0, x1, y1 = int(bbox[0]), int(bbox[1]), int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])
        # bbox_mask = mask[y0:y1+1, x0:x1+1]
        # unique_ids, counts = np.unique(bbox_mask, return_counts=True)
        # max_count_id = unique_ids[np.argmax(counts)]
        # binary_mask = np.where(mask == max_count_id, 1, 0)
        # self.save_mask_and_bbox(img_path=img_path, mask=binary_mask, bbox=bbox, output_path="./test_mask.png")
        # return binary_mask

        mask_path = img_path.replace("leftImg8bit", "gtFine")
        mask_path = os.path.splitext(mask_path)[0] + "_instanceIds.png"
        mask_path = os.path.join("data/cityscapes/", mask_path)
        # Load the mask for specific instance_id:
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)  # Load with original depth
        x0, y0, x1, y1 = int(bbox[0]), int(bbox[1]), int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])
        bbox_mask = mask[y0:y1+1, x0:x1+1]
        unique_ids = np.unique(bbox_mask)
        min_l2_dist = float('inf')
        chosen_id = None
        for uid in unique_ids:
            y, x = np.where(bbox_mask == uid)
            if len(x) == 0 or len(y) == 0:
                continue
            instance_bbox = [np.min(x), np.min(y), np.max(x), np.max(y)]
            l2_dist = np.sqrt(np.sum((np.array(instance_bbox) - np.array(bbox))**2))
            if l2_dist < min_l2_dist:
                min_l2_dist = l2_dist
                chosen_id = uid
        binary_mask = np.where(mask == chosen_id, 1, 0)
        self.save_mask_and_bbox(img_path=img_path, mask=binary_mask, bbox=bbox, output_path="./test_mask.png")
        return binary_mask
>>>>>>> be988ed87c5220278ea5289ef41ec11e78be4e23



    def get_category_mapping(self):
        self.cat_mapping = dict([i,i] for i in range(1, 21))

    def _filter_imgs(self):
        anns = self.coco.dataset['annotations']
        filtered_anns = []
        for ann in anns:
            if ann['bbox'][2] * ann['bbox'][3] > self.min_obj_size and ann['bbox'][2] * ann['bbox'][3] < self.max_obj_size and\
                ann['bbox'][2] > 2 and ann['bbox'][3] > 2:
                filtered_anns.append(ann)
        self.coco.dataset['annotations'] = filtered_anns

    def __len__(self):
        return len(self.coco.getAnnIds())
    
    def __getitem__(self, idx):
        ann = self.coco.dataset['annotations'][idx]
        img_info = self.coco.loadImgs(ann['image_id'])[0]
        h, w, file_name = img_info['height'], img_info['width'], img_info['file_name']
        img = self.get_image(file_name)

        # box mask
        bbox = ann['bbox']
        mask = np.zeros((h, w))
        x0, y0, x1, y1 = int(bbox[0]), int(bbox[1]), int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])
        mask[y0:y1+1, x0:x1+1] = 1
        
        ############################################################################################
        # Get GT Mask: 

        gt_mask = self.coco.annToMask(ann)

        ############################################################################################
        # Get depth: 

        depth_name = file_name.split("/")[-1].split(".")[0] + ".pt"
        depth_name = os.path.join("/workspace/mask-auto-labeler/data/cityscapes/depth_maps", depth_name)
        depth_map = torch.load(depth_name, map_location=torch.device('cpu'))
        
        ############################################################################################
        data = {'image': img, 'mask': mask, 'height': h, 'width': w, 
                'category_id': ann['category_id'], 'bbox': np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]], dtype=np.float32),
                'compact_category_id': self.cat_mapping[int(ann['category_id'])],
                'id': ann['id'], 'depth': depth_map, 'gt_mask': gt_mask}

        if self.transform is not None:
            data = self.transform(data)

        return data

    def get_image(self, file_name):
        file_name = "/".join(file_name.split('/')[2:])
        try:
            image = Image.open(os.path.join(self.img_data_dir, file_name)).convert('RGB')
            #image = image.resize((640, 480))

        except FileNotFoundError:
            return None
        return image


class BoxLabelVOCLMDB(Dataset):

    def __init__(self, ann_path, img_data_dir, min_obj_size=0, max_obj_size=1e12, transform=None, args=None):
        self.args = args
        self.ann_path = ann_path
        self.img_data_dir = img_data_dir
        self.min_obj_size = min_obj_size
        self.max_obj_size = max_obj_size
        self.transform = transform
        self.ann_env = lmdb.open(ann_path + "_ann", readonly=True, lock=False, create=False, subdir=True)
        self.img_env = lmdb.open(ann_path + "_image", readonly=True, lock=False, create=False, subdir=True)
        self._filter_imgs()
        self.get_category_mapping()

    def get_category_mapping(self):
        self.cat_mapping = dict([i,i] for i in range(1, 21))

    def _filter_imgs(self):
        self.mapping_ids = []
        with self.ann_env.begin() as txn:
            length = txn.stat()['entries']
            for idx in range(length):
                ann = json.loads(txn.get(str(idx).encode()))
                if ann['bbox'][2] * ann['bbox'][3] > self.min_obj_size and ann['bbox'][2] * ann['bbox'][3] < self.max_obj_size and\
                    ann['bbox'][2] > 2 and ann['bbox'][3] > 2:
                    self.mapping_ids.append(idx)
        self.__length__ = len(self.mapping_ids)


    def __len__(self):
        return self.__length__


    def __getitem__(self, _):
        idx = torch.randint(0, self.__length__, (1,))[0]
        with self.ann_env.begin(write=False) as txn:
            ann = json.loads(txn.get(str(self.mapping_ids[idx]).encode()))
        with self.img_env.begin(write=False) as txn:
            raw_img = txn.get(str(ann['image_id']).encode())
            if raw_img is None:
                return self[torch.randint(0, self.__length__, (1,))[0]]
            img = cv2.imdecode(
                np.fromstring(
                    pickle.loads(raw_img), np.uint8
                ), cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_info = pickle.loads(txn.get(str(ann['image_id']).encode()))

        h, w = img.shape[:2]

        # box mask
        mask = np.zeros((h, w))
        bbox = ann['bbox']
        x0, y0, x1, y1 = int(bbox[0]), int(bbox[1]), int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])
        mask[y0:y1+1, x0:x1+1] = 1

        data = {'image': img, 'mask': mask, 'height': h, 'width': w, 
                'category_id': ann['category_id'], 'bbox': np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]),
                'compact_category_id': self.cat_mapping[int(ann['category_id'])],
                'id': ann['id']}

        if self.transform is not None:
            data = self.transform(data)

        return data

    def get_image(self, file_name):
        try:
            image = Image.open(os.path.join(self.img_data_dir, file_name)).convert('RGB')
        except FileNotFoundError:
            return None
        return image




class InstSegVOC(BoxLabelVOC):

    def get_category_mapping(self):
        self.cat_mapping = dict([i,i] for i in range(1, 21))

    def __init__(self, *args, load_mask=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.args = args
        self.load_mask = load_mask
        self.get_category_mapping()

    def __getitem__(self, idx):
        ann = self.coco.dataset['annotations'][idx]
        img_info = self.coco.loadImgs(ann['image_id'])[0]
        h, w, file_name = img_info['height'], img_info['width'], img_info['file_name']
        img = self.get_image(file_name)
        # img = self.get_image(os.path.join(os.getcwd(),self.args[1], file_name))
        if img is None:
            return self[torch.randint(0, len(self), (1,))[0]]

        bbox = ann['bbox']
        x0, y0, x1, y1 = int(bbox[0]), int(bbox[1]), int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])

        # box mask
        boxmask = np.zeros((h, w))
        bbox = ann['bbox']
        x0, y0, x1, y1 = int(bbox[0]), int(bbox[1]), int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])
        boxmask[y0:y1+1, x0:x1+1] = 1


        data = {'image': img, 'boxmask': boxmask, 'height': h, 'width': w, 
                'category_id': ann['category_id'], 'bbox': np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]], dtype=np.float32),
                'compact_category_id': self.cat_mapping[int(ann['category_id'])],
                'id': ann['id'], 'image_id': ann['image_id']}

        if self.load_mask:
            mask = np.ascontiguousarray(maskUtils.decode(maskUtils.frPyObjects(ann['segmentation'], h, w)))
            if len(mask.shape) > 2:
                mask = mask.transpose((2, 0, 1)).sum(0) > 0
            mask = mask.astype(np.uint8)

            data['gtmask'] = DataWrapper(mask)
            data['mask'] = mask

        if self.transform is not None:
            data = self.transform(data)

        return data


class InstSegVOCwithBoxInput(InstSegVOC):

    def __init__(self, 
                 ann_path, 
                 img_data_dir, 
                 min_obj_size=0, 
                 max_obj_size=1e10, 
                 transform=None,
                 load_mask=True,
                 box_inputs=None):
        self.load_mask = load_mask
        self.ann_path = ann_path
        self.img_data_dir = img_data_dir
        self.min_obj_size = min_obj_size
        self.max_obj_size = max_obj_size
        self.transform = transform
        self.coco = COCO(ann_path)
        self._filter_imgs()
        self.get_category_mapping()
        with open(box_inputs, "r") as f:
            self.val_coco = json.load(f)
        self.__length__ = len(self.val_coco)
    
    def __len__(self):
        return self.__length__

    def __getitem__(self, idx):
        ann = self.val_coco[idx]
        img_info = self.coco.loadImgs(ann['image_id'])[0]
        h, w, file_name = img_info['height'], img_info['width'], img_info['file_name']
        img = self.get_image(file_name)

        # box mask
        boxmask = np.zeros((h, w))
        bbox = np.array(ann['bbox'])
        x0, y0, x1, y1 = int(bbox[0]), int(bbox[1]), int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])
        boxmask[y0:y1+1, x0:x1+1] = 1

        if 'id' not in ann.keys():
            _id = hash(str(ann['image_id']) + ' ' + str(x0) + ' ' + str(x1) + ' ' + str(y0) + ' ' + str(y1))
        else:
            _id = ann['id']
        
        data = {'image': img, 'boxmask': boxmask, 'height': h, 'width': w, 
                'category_id': ann['category_id'], 'bbox': np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]], dtype=np.float32),
                'compact_category_id': self.cat_mapping[int(ann['category_id'])],
                'id': _id, 'image_id': ann['image_id'], 'score': ann['score'] }

        if self.load_mask:
            mask = np.ascontiguousarray(maskUtils.decode(ann['segmentation']))
            if len(mask.shape) > 2:
                mask = mask.transpose((2, 0, 1)).sum(0) > 0
            mask = mask.astype(np.uint8)

            data['gtmask'] = DataWrapper(mask)
            data['mask'] = mask

        if self.transform is not None:
            data = self.transform(data)

        return data




class InstSegVOCLMDB(BoxLabelVOCLMDB):

    def get_category_mapping(self):
        self.cat_mapping = dict([i,i] for i in range(1, 21))

    def __init__(self, *args, load_mask=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.load_mask = load_mask
        self.get_category_mapping()

    def __getitem__(self, idx):
        with self.ann_env.begin() as txn:
            ann = json.loads(txn.get(str(self.mapping_ids[idx]).encode()))
        with self.img_env.begin() as txn:
            img_info = json.loads(txn.get(str(ann['image_id']).encode()))
        h, w, file_name = img_info['height'], img_info['width'], img_info['file_name']
        img = self.get_image(file_name)
        if img is None:
            return self[torch.randint(0, len(self), (1,))[0]]
        bbox = ann['bbox']
        x0, y0, x1, y1 = int(bbox[0]), int(bbox[1]), int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])

        # box mask
        boxmask = np.zeros((h, w))
        bbox = ann['bbox']
        x0, y0, x1, y1 = int(bbox[0]), int(bbox[1]), int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])
        boxmask[y0:y1+1, x0:x1+1] = 1

        data = {'image': img, 'height': h, 'width': w, 'boxmask': boxmask,
                'category_id': ann['category_id'], 'bbox': np.array([x0, y0, x1, y1]),
                'compact_category_id': self.cat_mapping[int(ann['category_id'])],
                'id': ann['id'], 'image_id': ann['image_id']}

        if self.load_mask:
            mask = np.ascontiguousarray(maskUtils.decode(maskUtils.frPyObjects(ann['segmentation'], h, w)))
            if len(mask.shape) > 2:
                mask = mask.transpose((2, 0, 1)).sum(0) > 0
            mask = mask.astype(np.uint8)

            data['gtmask'] = DataWrapper(mask)
            data['mask'] = mask

        if self.transform is not None:
            data = self.transform(data)

        return data

