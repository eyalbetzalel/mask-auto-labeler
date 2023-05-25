import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch

from prismer.experts.generate_depth import model
# from experts.depth.generate_dataset import Dataset, EyalDataset

def visualize_and_save_feature_map(feature_map, file_name):
    # Ensure the tensor is on the CPU
    feature_map = feature_map.cpu()
    feature_map = feature_map[0,:,:,:]
    # Normalize to [0,1]
    min_val = torch.min(feature_map)
    feature_map -= min_val
    max_val = torch.max(feature_map)
    feature_map /= max_val

    # If the feature map has more than 3 channels, we keep only the first 3
    if feature_map.shape[0] > 3:
        feature_map = feature_map[:3]
    # If it has only 1 channel, we duplicate it to have 3
    elif feature_map.shape[0] == 1:
        feature_map = torch.repeat(feature_map, 3, 1, 1)

    # Convert to numpy array
    feature_map_np = feature_map.numpy().transpose((1, 2, 0))

    # Plot feature map
    plt.imshow(feature_map_np, interpolation='nearest')
    plt.axis('off')

    # Save the figure
    plt.savefig(file_name, bbox_inches='tight', pad_inches=0)
    plt.close()
