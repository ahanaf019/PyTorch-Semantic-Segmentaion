import torch
import torch.nn as nn
from pathlib import Path


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