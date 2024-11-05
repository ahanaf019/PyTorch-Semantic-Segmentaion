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
import albumentations as A

from helpers import image_to_patches

matplotlib.use("GTK3Agg")


class BinSegDataset(Dataset):
    def __init__(self, image_paths, mask_paths, image_size, patch_size, num_classes, augment=False, augment_transforms=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.augment = augment
        
        self.augment_transforms = A.Compose([
            A.D4(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.5),
            A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
            A.XYMasking(num_masks_x=(0, 3), num_masks_y=(0, 3), mask_x_length=(10, 50), mask_y_length=(10, 50))
            # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
    
    
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
        
        if self.augment:
            transformed = self.augment_transforms(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
        
        
        image_patches = image_to_patches(image, self.patch_size) / 255.0
        mask_patches = image_to_patches(mask, self.patch_size)
        
        image_patches = torch.tensor(image_patches, dtype=torch.float32).permute(0, 3, 1, 2)
        mask_patches = torch.tensor(mask_patches, dtype=torch.int64)
        mask_patches = torch.nn.functional.one_hot(mask_patches.squeeze(), self.num_classes).permute(0, 3, 1, 2)
        return image_patches, mask_patches
    
    

    
    




def main():
    db_dir = Path('datasets/HRF')
    image_dir = db_dir / 'images'
    mask_dir = db_dir / 'manual1'
    
    image_paths = sorted(list(image_dir.glob('*')))
    mask_paths = sorted(list(mask_dir.glob('*')))
    
    ds = BinSegDataset(image_paths=image_paths, mask_paths=mask_paths, image_size=(224*10, 224*10), patch_size=224, num_classes=2)
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