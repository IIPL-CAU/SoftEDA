# Standard Library Modules
import argparse
# Pytorch Modules
import torch
from torch.optim.lr_scheduler import StepLR, LambdaLR, CosineAnnealingLR, CosineAnnealingWarmRestarts, ReduceLROnPlateau

def get_scheduler(optimizer: torch.optim.Optimizer, dataloader_length: int,
                  num_epochs: int=None, early_stopping_patience: int=None, learning_rate: float=None,
                  scheduler_type: str=None, args: argparse.Namespace=None) -> torch.optim.lr_scheduler:
    if num_epochs is None:
        if args is None:
            raise ValueError('Either num_epochs or args must be given.')
        else:
            num_epochs = args.num_epochs
    if early_stopping_patience is None:
        if args is None:
            raise ValueError('Either early_stopping_patience or args must be given.')
        else:
            early_stopping_patience = args.early_stopping_patience
    if learning_rate is None:
        if args is None:
            raise ValueError('Either learning_rate or args must be given.')
        else:
            learning_rate = args.learning_rate
    if scheduler_type is None:
        if args is None:
            raise ValueError('Either scheduler_type or args must be given.')
        else:
            scheduler_type = args.scheduler

    if scheduler_type == 'StepLR':
        epoch_step = num_epochs // 8 if num_epochs > 8 else 1
        return StepLR(optimizer, step_size=dataloader_length*epoch_step, gamma=0.8)
    elif scheduler_type == 'LambdaLR':
        lr_lambda = lambda epoch: 0.95 ** epoch
        return LambdaLR(optimizer, lr_lambda=lr_lambda)
    elif scheduler_type == 'CosineAnnealingLR':
        T_max = num_epochs // 8 if num_epochs > 8 else 1
        eta_min = learning_rate * 0.01
        return CosineAnnealingLR(optimizer, T_max=dataloader_length*T_max, eta_min=eta_min)
    elif scheduler_type == 'CosineAnnealingWarmRestarts':
        T_0 = num_epochs // 8 if num_epochs > 8 else 1
        T_mult = 2
        eta_min = learning_rate * 0.01
        return CosineAnnealingWarmRestarts(optimizer, T_0=dataloader_length*T_0,
                                           T_mult=dataloader_length*T_mult, eta_min=eta_min)
    elif scheduler_type == 'ReduceLROnPlateau':
        patience = early_stopping_patience // 2 if early_stopping_patience > 0 else num_epochs // 10 + 1
        return ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=patience)
    elif scheduler_type == 'None' or scheduler_type is None:
        return None
    else:
        raise ValueError(f'Unknown scheduler option {scheduler_type}')
