import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from typing import Optional, List, Dict
import os
from glob import glob



class BinSegDataset(Dataset):
    def __init__(self, image_paths, mask_paths):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
    
    
    def __len__(self):
        return len(self.image_paths)
    
    
    
    def __getitem__(self, index):
        print(len(self.image_paths))





def main():
    db_dir = Path('datasets/HRF')
    image_dir = db_dir / 'images'
    mask_dir = db_dir / 'manual1'
    
    image_paths = sorted(list(image_dir.glob('*')))
    mask_paths = sorted(list(mask_dir.glob('*')))
    
    for i, m in zip(image_paths, mask_paths):
        print(i, m)



if __name__ == "__main__":
    main()