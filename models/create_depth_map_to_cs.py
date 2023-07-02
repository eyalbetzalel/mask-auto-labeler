from torchvision.transforms.functional import pil_to_tensor
from prismer.experts.generate_depth import model as model_depth
model_depth = model_depth.cuda()
import os
import torch
import torchvision.transforms as transforms
from PIL import Image

def process_images(folder_path, model, output_file):
    # Create an empty dictionary to store the results
    depth_maps = {}

    # Iterate over the root folder and its subfolders
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
                    tensor_image = transforms.Resize((256, 512))(tensor_image)
                    tensor_image = tensor_image.cuda()
                    # Apply the depth estimation model
                    depth_map = model(tensor_image)

                    # Add the result to the dictionary
                    depth_maps[file_path] = depth_map

                except Exception as e:
                    print(f"Error processing file {file_path}: {str(e)}")

    # Save the dictionary to a file
    torch.save(depth_maps, output_file)

folder_path = "/workspace/mask-auto-labeler/data/cityscapes/leftImg8bit"
output_file = "/workspace/mask-auto-labeler/data/cityscapes/depth_maps.pt"
process_images(folder_path, model_depth, output_file)