import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch


import matplotlib

def visualize_and_save_feature_map(feature_map, segmentation_layer, file_name):
    # Ensure the tensors are on the CPU
    feature_map = feature_map.cpu()
    segmentation_layer = segmentation_layer.cpu()

    feature_map = feature_map[0,:,:,:]
    segmentation_layer = segmentation_layer[0,:,:]

    # Normalize to [0,1]
    min_val = torch.min(feature_map)
    feature_map -= min_val
    max_val = torch.max(feature_map)
    feature_map /= max_val

    # Normalize segmentation layer as well
    min_val = torch.min(segmentation_layer)
    segmentation_layer -= min_val
    max_val = torch.max(segmentation_layer)
    segmentation_layer /= max_val

    # If the feature map has more than 3 channels, we keep only the first 3
    if feature_map.shape[0] > 3:
        feature_map = feature_map[:3]
    # If it has only 1 channel, we duplicate it to have 3
    elif feature_map.shape[0] == 1:
        feature_map = torch.repeat(feature_map, 3, 1, 1)

    # Convert to numpy arrays
    feature_map_np = feature_map.numpy().transpose((1, 2, 0))
    segmentation_layer_np = segmentation_layer.numpy()
    # Plot feature map
    plt.imshow(feature_map_np, interpolation='nearest')
    
    # Plot segmentation layer over it with some transparency
    plt.imshow(segmentation_layer_np, interpolation='nearest', cmap='jet', alpha=0.5)

    plt.axis('off')

    # Save the figure
    plt.savefig(file_name, bbox_inches='tight', pad_inches=0)
    plt.close()


def visualize_and_save_depth_map(depth_map, file_name):
    # Ensure the tensor is on the CPU
    depth_map = depth_map.cpu()
    depth_map = depth_map[0,:,:]
    # Normalize to [0,1]
    min_val = torch.min(depth_map)
    depth_map -= min_val
    max_val = torch.max(depth_map)
    depth_map /= max_val

    # If the feature map has more than 3 channels, we keep only the first 3
    # if depth_map.shape[0] > 3:
    #     depth_map = depth_map[:3]
    # # If it has only 1 channel, we duplicate it to have 3
    # elif depth_map.shape[0] == 1:
    #     depth_map = torch.repeat(depth_map, 3, 1, 1)

    # Convert to numpy array
    depth_map_np = depth_map.numpy()

    # Plot feature map
    plt.imshow(depth_map_np, interpolation='nearest')
    plt.axis('off')

    # Save the figure
    plt.savefig(file_name, bbox_inches='tight', pad_inches=0)
    plt.close()

