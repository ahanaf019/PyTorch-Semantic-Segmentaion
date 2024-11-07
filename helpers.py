import torch
import torch.nn as nn
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import shutil

import torch.utils
import torch.utils.data


def save_state(save_path: Path, model: nn.Module, optim: torch.optim.Optimizer, epoch: int):
    model_state = model.state_dict()
    optim_state = optim.state_dict()
    state = {
        'epoch': epoch,
        'model_state': model_state,
        'optim_state': optim_state,
    }
    torch.save(state, save_path)
    print('State Saved.')
    

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





def plot_predictions(model: nn.Module, val_ds: torch.utils.data.DataLoader, 
                    plot_count: int, save_path: Path, epoch: int, image_size: int, patch_size: int, device: str):
    print('Generating figure...')
    itr = iter(val_ds)
    plt.figure(figsize=(plot_count * 4, plot_count * 4))
    model.eval()
    with torch.inference_mode():
        for i in range(plot_count):
            inputs, targets = next(itr)
            
            preds = []
            inputs_sub_batches = rebatch(inputs, batch_size=4)
            for img_batch in inputs_sub_batches:
                p = model(img_batch.to(device))
                p = torch.argmax(p, dim=1)
                preds.append(p)
            
            preds = torch.concat(preds)
            targets = torch.argmax(targets, dim=1)
            whole_image = patches_to_image(inputs.permute(0, 2, 3, 1), (image_size, image_size), patch_size)
            whole_mask = patches_to_image(targets.unsqueeze(1).permute(0, 2, 3, 1), (image_size, image_size), patch_size)
            whole_preds = patches_to_image(preds.unsqueeze(1).permute(0, 2, 3, 1), (image_size, image_size), patch_size)
            plt.subplot(plot_count,3,1+3*i)
            plt.imshow(whole_image)
            plt.title('image')
            plt.axis("off")
            plt.subplot(plot_count,3,2+3*i)
            plt.imshow(whole_mask.cpu(), cmap='gray')
            plt.title('mask')
            plt.axis("off")
            plt.subplot(plot_count,3,3+3*i)
            plt.imshow(whole_preds.cpu(), cmap='gray')
            plt.title('pred_mask')
            plt.axis("off")
        plt.tight_layout()
        
    if not save_path.exists():
        save_path.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(save_path / f'best-{epoch}.png')
    plt.close()