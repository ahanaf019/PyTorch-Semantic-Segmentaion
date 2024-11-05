import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from torchmetrics import Accuracy, JaccardIndex
from dataset import BinSegDataset
from helpers import rebatch, plot_predictions
from models import UNet
from torchsummary import summary
from helpers import save_state
from torch.utils.data import DataLoader
import os
from typing import Callable


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(torch.__version__)
print(f'Using device: {device}')


TRAIN_SPLIT = 0.85
VAL_SPLIT = 0.1
PATCH_SIZE = 224
PATCH_COUNT = 10
IMAGE_SIZE = PATCH_SIZE * PATCH_COUNT
N_FEATURES = 32
NUM_CLASSES = 2
BATCH_SIZE = 32
NUM_EPOCHS = 100
LEARNING_RATE = 5e-4
PLOT_COUNT = 4


accuracy_score = Accuracy(task='multiclass', num_classes=2).to(device)
jaccard_score = JaccardIndex(task='multiclass', num_classes=2).to(device)

def main():
    db_name = 'HRF'
    db_dir = Path('datasets')
    
    image_dir = db_dir / db_name / 'images'
    mask_dir = db_dir / db_name / 'manual1'
    
    image_paths = sorted(list(image_dir.glob('*')))
    mask_paths = sorted(list(mask_dir.glob('*')))
    print(f'Total images and masks: {len(image_paths)}, {len(mask_paths)}')
    
    train_images, test_images, train_masks, test_masks = train_test_split(
        image_paths, mask_paths, train_size=TRAIN_SPLIT, shuffle=True, random_state=42
    )
    
    train_images, val_images, train_masks, val_masks = train_test_split(
        train_images, train_masks, test_size=VAL_SPLIT, shuffle=True, random_state=42
    )
    
    print(f'Total train images and masks: {len(train_images)}, {len(train_masks)}')
    print(f'Total val images and masks: {len(val_images)}, {len(val_masks)}')
    print(f'Total test images and masks: {len(test_images)}, {len(test_masks)}')
    
    train_ds = BinSegDataset(train_images, train_masks, (IMAGE_SIZE, IMAGE_SIZE), PATCH_SIZE, NUM_CLASSES, augment=True)
    val_ds = BinSegDataset(val_images, val_masks, (IMAGE_SIZE, IMAGE_SIZE), PATCH_SIZE, NUM_CLASSES)
    
    train_loader = DataLoader(train_ds, batch_size=None, shuffle=True, num_workers=os.cpu_count(), prefetch_factor=1)
    val_loader = DataLoader(val_ds, batch_size=None, shuffle=False)
    
    print(len(train_ds))
    img, msk = next(iter(train_ds))
    print(img.shape, torch.min(img), torch.max(img))
    # plt.subplot(1,2,1)
    # plt.imshow(img[7].permute(1, 2, 0))
    # plt.subplot(1,2,2)
    # plt.imshow(torch.argmax(msk[7].permute(1, 2, 0), dim=-1), cmap='gray')
    # plt.show()
    
    model = UNet(3, N_FEATURES, NUM_CLASSES).to(device)
    summary(model, input_size=(3, PATCH_SIZE, PATCH_SIZE), batch_size=BATCH_SIZE)
    
    optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()
    
    save_dir = Path('models')
    save_path = save_dir / 'unet.pth'
    history = train_model(model, train_loader, val_loader, save_path, NUM_EPOCHS, optim, loss_fn, NUM_CLASSES)
    




from tqdm.auto import tqdm

def train_step(model: nn.Module, train_ds: DataLoader, loss_fn: Callable, optim: torch.optim.Optimizer, num_classes: int):
    accs = []
    ious = []
    losses = []
    
    
    for inputs, targets in tqdm(train_ds):
        inputs = inputs.to(device)
        targets = targets.type(torch.float32).to(device)
        # targets_onehot = nn.functional.one_hot(targets, num_classes).type(torch.float32)
        # print(targets_onehot.shape)
        
        inputs_sub_batches = rebatch(inputs, batch_size=BATCH_SIZE)
        targets_sub_batches = rebatch(targets, batch_size=BATCH_SIZE)
        
        
        for img_batch, mask_batch in zip(inputs_sub_batches, targets_sub_batches):
            # print(img_batch.shape, mask_batch.shape)
            model.train()
            preds = model(img_batch)
            loss = loss_fn(preds, mask_batch)
            
            optim.zero_grad()
            loss.backward()
            optim.step()
            
            # print(torch.argmax(preds, dim=1).shape, torch.argmax(mask_batch, dim=1).shape)
            # print(preds.device, mask_batch.device)
            accs.append(accuracy_score(torch.argmax(preds, dim=1), torch.argmax(mask_batch, dim=1)).cpu().numpy())
            ious.append(jaccard_score(torch.argmax(preds, dim=1), torch.argmax(mask_batch, dim=1)).cpu().numpy())
            losses.append(loss.detach().cpu().numpy())
        
    return np.mean(losses), np.mean(accs), np.mean(ious)


def validation_step(model: nn.Module, val_ds: DataLoader, loss_fn, num_classes: int):
    accs = []
    losses = []
    ious = []
    
    model.eval()
    with torch.inference_mode():
        for inputs, targets in tqdm(val_ds):
            inputs = inputs.to(device)
            targets = targets.type(torch.float32).to(device)
            
            inputs_sub_batches = rebatch(inputs, batch_size=BATCH_SIZE)
            targets_sub_batches = rebatch(targets, batch_size=BATCH_SIZE)
            
            for img_batch, mask_batch in zip(inputs_sub_batches, targets_sub_batches):
                preds = model(img_batch)
                loss = loss_fn(preds, mask_batch)
                
                accs.append(accuracy_score(torch.argmax(preds, dim=1), torch.argmax(mask_batch, dim=1)).cpu().numpy())
                ious.append(jaccard_score(torch.argmax(preds, dim=1), torch.argmax(mask_batch, dim=1)).cpu().numpy())
                losses.append(loss.detach().cpu().numpy())
                
    return np.mean(losses), np.mean(accs), np.mean(ious)



def train_model(
    model: nn.Module, 
    train_ds: DataLoader, val_ds: DataLoader,
    save_path: Path, num_epochs: int, optim: torch.optim.Optimizer, loss_fn, num_classes: int):
    
    losses = []
    accs = []
    ious = []
    
    val_losses = []
    val_accs = []
    val_ious = []
    
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', factor=0.5, patience=15)
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}:')
        
        train_loss, train_acc, train_iou = train_step(model, train_ds, loss_fn, optim, num_classes)
        val_loss, val_acc, val_iou = validation_step(model, val_ds, loss_fn, num_classes)
        
        lr_scheduler.step(val_loss)
        last_lr = lr_scheduler.get_last_lr()
        
        print(f'loss: {train_loss:0.4f} | acc: {train_acc:0.4f} | iou: {train_iou:0.4f} | val_loss: {val_loss:0.4f} | val_acc: {val_acc:0.4f} | val_iou: {val_iou:0.4f} | lr: {last_lr[0]:0.6f}\n')
        
        if len(val_losses) == 0 or val_loss < np.min(val_losses):
            save_state(save_path, model, optim, epoch)
            
            plot_predictions(model, val_ds, PLOT_COUNT, Path('figs'), epoch, IMAGE_SIZE, PATCH_SIZE, device)
            
        
        
        losses.append(train_loss)
        accs.append(train_acc)
        ious.append(train_iou)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        val_ious.append(val_iou)
        
    return {
        'loss': losses,
        'acc': accs,
        'iou': ious,
        'val_loss': val_losses,
        'val_iou': val_ious,
        'val_acc': val_accs
    }




if __name__ == "__main__":
    main()