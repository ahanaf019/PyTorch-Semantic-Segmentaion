import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from typing import Optional, List, Dict
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from glob import glob

matplotlib.use("GTK3Agg")


class BinSegDataset(Dataset):
    def __init__(self, image_paths, mask_paths, image_size, patch_size):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.image_size = image_size
        self.patch_size = patch_size
    
    
    def __len__(self):
        return len(self.image_paths)
    
    
    
    def __getitem__(self, index):
        clahe = cv2.createCLAHE(clipLimit=5, tileGridSize=(12, 12))
        
        
        image = cv2.imread(self.image_paths[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.image_size)
        image[:, :, 0] = clahe.apply(image[:, :, 0])
        image[:, :, 1] = clahe.apply(image[:, :, 1])
        image[:, :, 2] = clahe.apply(image[:, :, 2])
        
        mask = cv2.imread(self.mask_paths[index])
        mask = cv2.resize(mask, self.image_size, interpolation=cv2.INTER_NEAREST)
        mask = mask[:, :, 0]
        mask = mask / 255.0
        # print(image.shape, mask.shape)
        
        image_patches = image_to_patches(image, self.patch_size)
        mask_patches = image_to_patches(mask, self.patch_size)
        
        image_patches = torch.tensor(image_patches, dtype=torch.float32).permute(0, 3, 1, 2)
        mask_patches = torch.tensor(mask_patches, dtype=torch.float32).permute(0, 3, 1, 2)
        return image_patches, mask_patches
    
    
def image_to_patches(image, p_size):
    """
    Splits an image into patches of size p_size x p_size, padding the image if necessary.
    
    Parameters:
    image (numpy array): Input image of shape (H, W, C) or (H, W).
    p_size (int): Size of the patches to extract (p_size x p_size).
    
    Returns:
    tuple: (Padded image, Patches in shape (num_patches_y, num_patches_x, p_size, p_size, C) 
            if 3D input, else (num_patches_y, num_patches_x, p_size, p_size))
    """
    h, w = image.shape[:2]
    
    # Calculate the padding needed to make the dimensions divisible by p_size
    pad_h = (p_size - h % p_size) % p_size
    pad_w = (p_size - w % p_size) % p_size
    
    # Pad the image with zeros (or other values if necessary)
    if len(image.shape) == 3:  # RGB or multi-channel image
        image_padded = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')
    else:  # Grayscale image
        image_padded = np.pad(image, ((0, pad_h), (0, pad_w)), mode='constant')
    
    # Get new dimensions after padding
    h_padded, w_padded = image_padded.shape[:2]
    
    # Split the image into patches
    patches = image_padded.reshape(h_padded // p_size, p_size, w_padded // p_size, p_size, -1)
    patches = patches.swapaxes(1, 2)  # Move patch dimensions together
    
    num_patches_y, num_patches_x = patches.shape[:2]
    patches = patches.reshape(num_patches_y * num_patches_x, p_size, p_size, -1)
    
    return patches


def patches_to_image(patches, image_shape, p_size):
    """
    Reconstructs the original (padded) image from patches.
    
    Parameters:
    patches (numpy array): Patches of shape (num_patches, p_size, p_size, C) or (num_patches, p_size, p_size).
    image_shape (tuple): Shape of the original padded image (H, W, C) or (H, W).
    p_size (int): Size of the patches (p_size x p_size).
    
    Returns:
    numpy array: Reconstructed image of shape image_shape.
    """
    h_padded, w_padded = image_shape[:2]
    
    # Calculate the number of patches along height and width
    num_patches_y = h_padded // p_size
    num_patches_x = w_padded // p_size
    
    # Reshape patches back into the grid shape (num_patches_y, num_patches_x, p_size, p_size, C)
    patches = patches.reshape(num_patches_y, num_patches_x, p_size, p_size, -1)
    
    # Swap axes back to combine patches into the original padded image
    patches = patches.swapaxes(1, 2)
    
    # Reshape to the original image size
    reconstructed_image = patches.reshape(h_padded, w_padded, -1)
    
    return reconstructed_image
    
    
def rebatch(patches, batch_size):
    num_patches = patches.shape[0]
    # Split into sub-batches of the desired size
    sub_batches = [patches[i:i + batch_size] for i in range(0, num_patches, batch_size)]
    return sub_batches




def main():
    db_dir = Path('datasets/HRF')
    image_dir = db_dir / 'images'
    mask_dir = db_dir / 'manual1'
    
    image_paths = sorted(list(image_dir.glob('*')))
    mask_paths = sorted(list(mask_dir.glob('*')))
    
    ds = BinSegDataset(image_paths=image_paths, mask_paths=mask_paths, image_size=(224*10, 224*10), patch_size=224)
    dl = DataLoader(ds, batch_size=None, shuffle=True)
    x, y = next(iter(dl))
    print(x.shape)
    
    # print(len(ds))
    # for image, mask in ds:
    #     print(image.shape, mask.shape)
    #     plt.subplot(1,2,1)
    #     plt.imshow(image[2])
    #     plt.subplot(1,2,2)
    #     plt.imshow(mask[2], cmap='gray')
    #     plt.show()
    #     break



if __name__ == "__main__":
    main()