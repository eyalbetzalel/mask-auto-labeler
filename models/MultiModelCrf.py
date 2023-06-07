import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


def visualize_and_save_feature_map(feature_map, segmentation_layer, file_name):

    # get colormap
    ncolors = 256
    color_array = plt.get_cmap('gist_rainbow')(range(ncolors))

    # change alpha values
    color_array[:,-1] = np.linspace(0.0,1.0,ncolors)

    # create a colormap object
    map_object = LinearSegmentedColormap.from_list(name='rainbow_alpha',colors=color_array)

    # register this new colormap with matplotlib
    plt.register_cmap(cmap=map_object)
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
    plt.imshow(segmentation_layer_np, interpolation='none', cmap='rainbow_alpha', alpha=0.5)

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

def visualize_and_save_batch(feature_map, segmentation_layer, base_file_name, text=None, plot_orig=True):


    # get colormap
    ncolors = 256
    color_array = plt.get_cmap('gist_rainbow')(range(ncolors))

    # change alpha values
    color_array[:,-1] = np.linspace(0.0,1.0,ncolors)

    # create a colormap object
    map_object = LinearSegmentedColormap.from_list(name='rainbow_alpha',colors=color_array)

    # register this new colormap with matplotlib
    plt.register_cmap(cmap=map_object)
    
    # Ensure the tensors are on the CPU
    feature_map = feature_map.cpu()
    segmentation_layer = segmentation_layer.cpu()

    # Go through each image in the batch
    for i in range(feature_map.shape[0]):
        # Prepare image and segmentation map
        single_feature_map = feature_map[i,:,:,:]
        single_segmentation_layer = segmentation_layer[i,:,:]

        # Normalize to [0,1]
        min_val = torch.min(single_feature_map)
        single_feature_map -= min_val
        max_val = torch.max(single_feature_map)
        single_feature_map /= max_val

        min_val = torch.min(single_segmentation_layer)
        single_segmentation_layer -= min_val
        max_val = torch.max(single_segmentation_layer)
        single_segmentation_layer /= max_val

        # Convert to numpy arrays
        feature_map_np = single_feature_map.numpy().transpose((1, 2, 0))
        segmentation_layer_np = single_segmentation_layer.numpy()

        if plot_orig:
            # Save image without segmentation layer
            plt.imshow(feature_map_np, interpolation='nearest')
            plt.axis('off')
            plt.savefig(base_file_name + f"_image_{i}.png", bbox_inches='tight', pad_inches=0)
            plt.close()

        # Save image with segmentation layer
        plt.imshow(feature_map_np, interpolation='nearest')
        plt.imshow(segmentation_layer_np, interpolation='none', cmap='rainbow_alpha', alpha=0.5)
        if text is not None:
            plt.text(0.95, 0.05, text, ha='right', va='bottom', transform=plt.gca().transAxes)
        plt.axis('off')
        plt.savefig(base_file_name + f"_image_with_segment_{i}.png", bbox_inches='tight', pad_inches=0)
        plt.close()

import matplotlib.gridspec as gridspec

import matplotlib.gridspec as gridspec

import matplotlib.gridspec as gridspec

def visualize_and_save_all(feature_map, seg_original, seg_rgb, seg_depth, depth_map, base_file_name):

    # get colormap
    ncolors = 256
    color_array = plt.get_cmap('gist_rainbow')(range(ncolors))

    # change alpha values
    color_array[:,-1] = np.linspace(0.0,1.0,ncolors)

    # create a colormap object
    map_object = LinearSegmentedColormap.from_list(name='rainbow_alpha',colors=color_array)

    # register this new colormap with matplotlib
    plt.register_cmap(cmap=map_object)

    # Ensure the tensors are on the CPU
    feature_map = feature_map.cpu()
    seg_original = seg_original.cpu()
    seg_rgb = seg_rgb.cpu()
    seg_depth = seg_depth.cpu()
    depth_map = depth_map.cpu()

    for i in range(feature_map.shape[0]):
        fig = plt.figure(figsize=(10,10))
        gs = gridspec.GridSpec(2, 3)

        # Normalize feature map to [0,1]
        single_feature_map = feature_map[i,:,:,:]
        min_val = torch.min(single_feature_map)
        single_feature_map -= min_val
        max_val = torch.max(single_feature_map)
        single_feature_map /= max_val

        # Convert to numpy array
        feature_map_np = single_feature_map.numpy().transpose((1, 2, 0))

        # Image without segmentation
        ax = fig.add_subplot(gs[0, 0])
        ax.imshow(feature_map_np, interpolation='nearest')
        ax.axis('off')
        ax.text(0.05, 0.95, 'Image', ha='left', va='top', transform=ax.transAxes, color='white')

        for j, (seg_layer, title) in enumerate(zip([seg_original, seg_rgb, seg_depth], ['Original', 'RGB', 'Depth']), start=1):
            # Normalize segmentation layer to [0,1]
            single_segmentation_layer = seg_layer[i,:,:]
            min_val = torch.min(single_segmentation_layer)
            single_segmentation_layer -= min_val
            max_val = torch.max(single_segmentation_layer)
            single_segmentation_layer /= max_val

            # Convert to numpy array
            segmentation_layer_np = single_segmentation_layer.numpy()

            # Image with segmentation layer
            ax = fig.add_subplot(gs[j//3, j%3])
            ax.imshow(feature_map_np, interpolation='nearest')
            ax.imshow(segmentation_layer_np, interpolation='none', cmap='rainbow_alpha', alpha=0.5)
            ax.text(0.05, 0.95, title, ha='left', va='top', transform=ax.transAxes, color='white')
            ax.axis('off')

        # Depth map
        single_depth_map = depth_map[i,:,:]
        min_val = torch.min(single_depth_map)
        single_depth_map -= min_val
        max_val = torch.max(single_depth_map)
        single_depth_map /= max_val
        depth_map_np = single_depth_map.numpy()

        ax = fig.add_subplot(gs[1, 2])
        ax.imshow(depth_map_np, interpolation='nearest')
        ax.text(0.05, 0.95, 'Depth', ha='left', va='top', transform=ax.transAxes, color='white')
        ax.axis('off')

        plt.subplots_adjust(wspace=0.01, hspace=0.01) # reduce spacing
        plt.tight_layout(pad=0.5) # Here we set the padding to a smaller value to get the subplots closer


        plt.savefig(base_file_name + f"_combined_{i}.png", bbox_inches='tight', pad_inches=0)
        plt.close()
