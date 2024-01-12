import os
from PIL import Image
import numpy as np
import cv2 
import json
import matplotlib.pyplot as plt
import torch
from segment_anything import sam_model_registry, SamPredictor
from tqdm import tqdm
from torch.nn import functional as F



def calculate_distances(feature_tensor, pixel_coord, distance_method='euclidean'):
    """
    Calculate the distances between the feature vector at the specified pixel coordinate
    and all other feature vectors in the tensor.

    Args:
    - feature_tensor (numpy.ndarray): The feature tensor with shape [32, 1024, 2048].
    - pixel_coord (tuple): The (X, Y) coordinate of the pixel.
    - distance_method (str): The method used to calculate distances ('euclidean', 'manhattan', etc.)

    Returns:
    - numpy.ndarray: An array of distances with shape [1024, 2048].
    """
    if distance_method == 'euclidean':
        distance_func = lambda a, b: np.sqrt(np.sum((a - b) ** 2))
    elif distance_method == 'manhattan':
        distance_func = lambda a, b: np.sum(np.abs(a - b))
    else:
        raise ValueError("Unsupported distance method")

    x, y = pixel_coord
    target_vector = feature_tensor[:, x, y]

    distances = np.zeros((feature_tensor.shape[1], feature_tensor.shape[2]))

    for i in range(feature_tensor.shape[1]):
        for j in range(feature_tensor.shape[2]):
            distances[i, j] = distance_func(target_vector, feature_tensor[:, i, j])

    return distances

def visualize_distances(image_path, distances, pixel_coord, output_path):
    """
    Visualize the distances as a heatmap overlaid on the original image and save the result.

    Args:
    - image_path (str): The path to the image file.
    - distances (numpy.ndarray): The distances array with shape [1024, 2048].
    - pixel_coord (tuple): The (X, Y) coordinate of the pixel.
    - output_path (str): The path to save the output image.
    """
    # Load the image
    image = Image.open(image_path)
    plt.imshow(image, extent=[0, distances.shape[1], distances.shape[0], 0])

    # Overlay the heatmap
    plt.imshow(distances, cmap='hot', alpha=0.5, extent=[0, distances.shape[1], distances.shape[0], 0])

    # Mark the input pixel coordinate
    plt.scatter(*pixel_coord, color='blue', s=50)

    # Save the result
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()



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
    
    # val_flag = False

    # # Get image : 
     
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

    #################################################################################################
    # image embedding :
    img_embedding = predictor.get_image_embedding()
    img_embedding = F.interpolate(
        img_embedding,
        (1024, 1024),
        mode="bilinear",
        align_corners=False,
    )
    img_embedding = img_embedding[..., : 512, : 1024]
    img_embedding = F.interpolate(img_embedding, (1024, 2048), mode="bilinear", align_corners=False)
    img_embedding = img_embedding.squeeze().cpu().numpy()
    #################################################################################################

    # loop predicted bounding boxes:
    
    mask = None

    for box in bbox:

        box = np.round(np.array(box)).astype(int)

        # get mask for each bounding box
        
        mask, _, _, features = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=box[None, :],
        multimask_output=False,)

        mask = mask.astype(np.uint8).squeeze()

        # convert mask to polygon
        polygon = mask_to_polygon(mask)
        polygons.append(polygon)
    
    return polygons, mask, features, img_embedding

# Update json file with list of polygons:

def sam_jason():
        
        val_flag = True

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

def sam_heatmap():
        
        val_flag = True

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
                        annotation_id += 1
                        if annotation_id <= 40:
                            continue
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
                        
                        if len(bbox) == 0:
                            continue

                        polygons, mask, features, img_embedding = sam_get_masks(bbox, img_path)

                        # Example usage
                        pixel_coord = (512, 512)  # Replace with actual pixel coordinate
                        output_path = f'sam_output_with_bbox_prompt_{annotation_id}.png'  # Replace with the desired output path
                        distances = calculate_distances(features, pixel_coord)
                        visualize_distances(img_path, distances, pixel_coord, output_path)

                        output_path = f'sam_output_img_emb_{annotation_id}.png'  # Replace with the desired output path
                        distances = calculate_distances(img_embedding, pixel_coord)
                        visualize_distances(img_path, distances, pixel_coord, output_path)
                        v=0



                        





                    



if __name__ == '__main__':
    sam_heatmap()

