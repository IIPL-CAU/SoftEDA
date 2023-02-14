# Standard Library Modules
import argparse
# Pytorch Modules
import torch
import torch.nn as nn

def get_optimizer(model: nn.Module, learning_rate: float=None, weight_decay: float=None,
                  optim_type: str=None, args: argparse.Namespace=None) -> torch.optim.Optimizer:
    if learning_rate is None:
        if args is None:
            raise ValueError('Either learning_rate or args must be given.')
        else:
            learning_rate = args.learning_rate
    if weight_decay is None:
        if args is None:
            raise ValueError('Either weight_decay or args must be given.')
        else:
            weight_decay = args.weight_decay
    if optim_type is None:
        if args is None:
            raise ValueError('Either optim_type or args must be given.')
        else:
            optim_type = args.optimizer

    if weight_decay > 0:
        if optim_type == 'SGD':
            return torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optim_type == 'AdaDelta':
            return torch.optim.Adadelta(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optim_type == 'Adam':
            return torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optim_type == 'AdamW':
            return torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        else:
            raise ValueError(f'Unknown optimizer option {optim_type}')
    else:
        if optim_type == 'SGD':
            return torch.optim.SGD(model.parameters(), lr=learning_rate)
        elif optim_type == 'Adam':
            return torch.optim.Adam(model.parameters(), lr=learning_rate)
        elif optim_type == 'AdamW':
            return torch.optim.AdamW(model.parameters(), lr=learning_rate)
        else:
            raise ValueError(f'Unknown optimizer option {optim_type}')
