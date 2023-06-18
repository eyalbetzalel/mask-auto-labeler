import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

# Cityscapes imports 
from cityscapesscripts.helpers.labels import labels, name2label, id2label

CLASS_NAMES = [l.name for l in labels if l.hasInstances]
CLASS2IDX = {l: name2label[l].id for l in CLASS_NAMES}

def list_all_files_that_ends_with(path, ends=""):  # path is a directory
    files = []
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in [f for f in filenames if f.endswith(ends)]:
            files.append(os.sep.join([dirpath, filename]))
    return files

class BoxLabelCityscapes(Dataset):
    def __init__(self, ann_path, img_data_dir, min_obj_size=0, max_obj_size=1e10, load_mask=False, transform=None, args=None):
        self.img_data_dir = img_data_dir
        self.ann_path = ann_path
        self.min_obj_size = min_obj_size
        self.max_obj_size = max_obj_size
        self.transform = transform
        self.img_paths = sorted(list_all_files_that_ends_with(path=img_data_dir, ends=".png"))
        self.ann_paths = sorted(list_all_files_that_ends_with(path=ann_path, ends="gtFine_instanceIds.png"))
        v=0
        #self.img_paths = sorted(glob.glob(os.path.join(img_data_dir, "*.png")))
        #self.ann_paths = sorted(glob.glob(os.path.join(ann_path, "*gtFine_instanceIds.png")))

    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        ann_path = self.ann_paths[idx]

        img = ToTensor()(Image.open(img_path).convert('RGB'))
        img = img.permute(1, 2, 0)
        instances = np.asarray(Image.open(ann_path))

        unique_ids = np.unique(instances)
        instance_ids = [id for id in unique_ids if id >= 24]  # 24 is the minimum id for objects with instance-level annotations

        bboxes = []
        masks = []
        labels_arr = []
        for instance_id in instance_ids:
            instance_mask = instances == instance_id
            y, x = np.where(instance_mask)
            bboxes.append([x.min(), y.min(), x.max(), y.max()])  # compute bounding box from instance mask
            masks.append(instance_mask)
            if instance_id // 1000 == 0:
                instance_id = instance_id * 1000
            labels_arr.append(instance_id // 1000)

        data = {'image': img, 'bbox': bboxes, 'mask': masks, 'label': labels_arr}

        if self.transform:
            data = self.transform(data)
            
        return data

