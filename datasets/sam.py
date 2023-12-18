import os
from PIL import Image
import numpy as np
import cv2 
import json
import matplotlib.pyplot as plt
import torch
from segment_anything import sam_model_registry, SamPredictor
from tqdm import tqdm


# Import SAM model:

def setup_sam_model():
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device="cuda:1")
    predictor = SamPredictor(sam)
    return predictor

predictor = setup_sam_model()


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    

def get_image(file_name):
    try:
        image = Image.open(file_name).convert('RGB')
        #image = image.resize((640, 480))

    except FileNotFoundError:
        return None
    return image

def mask_to_polygon(binary_mask):
    """Convert binary mask to polygons"""
    # Find contours in the binary mask
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polygons = []
    for contour in contours:
        # Make sure contour area is large enough
        if cv2.contourArea(contour) > 2:
            # Flatten and append to the polygons list
            polygons.append(contour.flatten().tolist())
    
    return polygons
  
def sam_get_masks(bbox, path):
    
    # Get image : 
     
    # if val_flag:
    #     img_path = path.replace("/maskdino_labels/", "/")
    # else:
    #     img_path = path.replace("/maskdino_labels_no_gt/", "/")
    # img_path = os.path.splitext(img_path)[0] + ".png"

    #img_path = "/workspace/mask-auto-labeler/data/cityscapes/leftImg8bit/train/aachen/aachen_000000_000019_leftImg8bit.png"
    # Prepare to recreate the Instances object
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    predictor.set_image(img)

    polygons = []

    # loop predicted bounding boxes:

    for box in bbox:

        box = np.round(np.array(box)).astype(int)

        # get mask for each bounding box
        
        mask, _, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=box[None, :],
        multimask_output=False,)

        mask = mask.astype(np.uint8).squeeze()

        # convert mask to polygon
        polygon = mask_to_polygon(mask)
        polygons.append(polygon)
    
    return polygons

# Update json file with list of polygons:


def mock():
    path = "/workspace/mask-auto-labeler/data/cityscapes/maskdino_labels/leftImg8bit/train/aachen/aachen_000000_000019_leftImg8bit.json"
    with open(path, 'r') as f:
        data = json.load(f)
    bbox = data['pred_boxes']
    val_flag = False

    plt.figure(figsize=(10, 10))
    
    firstVisible = True
    # mask, box, img = sam_get_masks(bbox, path, val_flag)
    # Iterate over all bounding boxes
    for single_box in bbox:
        mask, box, img = sam_get_masks([single_box], path, val_flag)  # Process one box at a time
        if firstVisible:
            firstVisible = False
            plt.imshow(img)
        show_mask(mask, plt.gca())
        show_box(box, plt.gca())
    

    plt.axis('off')
    plt.savefig('test.png')

def sam_jason():
        
        val_flag = False

        if val_flag:
            # This is validation set after GT filtering
            folder_path = '/workspace/mask-auto-labeler/data/cityscapes/maskdino_labels'
            folder_path = folder_path + "/leftImg8bit/val"
        else:
            # This is training set before GT filtering (MAskDINO Output)
            folder_path = '/workspace/mask-auto-labeler/data/cityscapes/maskdino_labels_no_gt'
            folder_path = folder_path + "/leftImg8bit/train"


        # List to store all annotations
        annotations = []
        
        # Running index for annotations across all JSON files
        annotation_id = 0
        
        # Traverse through all JSON files in the folder
        for dirpath, dirnames, filenames in os.walk(folder_path):
            for filename in tqdm(filenames, desc="Processing JSON files"):
                if filename.endswith(".json"):
                    with open(os.path.join(dirpath, filename), 'r') as f:
                        data = json.load(f)
                        bbox = data['pred_boxes']
                        # bbox is list of lists that contains bounding box coordinates in floating point. change to int:
                        bbox = np.round(np.array(bbox)).astype(int)
                        # change bbox to list of lists:
                        bbox = bbox.tolist()
                        # Get image path:
                        
                        fullpath = os.path.join(dirpath, filename)
                        if val_flag:
                            img_path = fullpath.replace("/maskdino_labels/", "/")
                        else:
                            img_path = fullpath.replace("/maskdino_labels_no_gt/", "/")
                        img_path = os.path.splitext(img_path)[0] + ".png"

                        polygons = sam_get_masks(bbox, img_path)
                        data["sam_seg"] = polygons
                        v=0
                    
                    with open(os.path.join(dirpath, filename), 'w') as f:
                        json.dump(data, f, indent=4)
                        v=0




if __name__ == '__main__':
    sam_jason()