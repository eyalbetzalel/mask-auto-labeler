from torchvision.transforms.functional import pil_to_tensor
from prismer.experts.generate_depth import model as model_depth
model_depth = model_depth.cuda(0)
import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from MultiModelCrf import visualize_and_save_feature_map, visualize_and_save_depth_map, visualize_and_save_batch, visualize_and_save_all
from tqdm import tqdm

def process_images(folder_path, model, output_file):
    depth_maps = {}
    count = 0
    total_images = sum(len(files) for _, _, files in os.walk(folder_path))

    # Iterate over the root folder and its subfolders
    with tqdm(total=total_images, desc='Processing Images') as pbar:
        for root, dirs, files in os.walk(folder_path):
            for file_name in files:
                # Check if the file is a PNG image
                
                if file_name.endswith('.png'):
                    file_path = os.path.join(root, file_name)
                    try:
                        # Load the image using PIL
                        image = Image.open(file_path)
                        # Transform the image into a tensor
                        transform = transforms.ToTensor()
                        tensor_image = transform(image).unsqueeze(0)
                        tensor_image = transforms.Resize((512, 1024))(tensor_image)
                        tensor_image = tensor_image.cuda(0)
                        # Apply the depth estimation model
                        depth_map = model(tensor_image)
                        depth_map = transforms.Resize((1024, 2048))(depth_map)
                        depth_maps[file_path] = depth_map.detach().cpu()
                        
                        if count % 1000 == 0 :
                            torch.save(depth_maps, output_file)
                            v=0
                        count += 1
                    except Exception as e:
                        print(f"Error processing file {file_path}: {str(e)}")
                pbar.update(1)
        torch.save(depth_maps, output_file)

folder_path = "/workspace/mask-auto-labeler/data/cityscapes/leftImg8bit"
output_file = "/workspace/mask-auto-labeler/data/cityscapes/depth_maps.pt"
process_images(folder_path, model_depth, output_file)