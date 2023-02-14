# Standard Library Modules
import os
import sys
import time
import tqdm
import random
import logging
import argparse
# 3rd-party Modules
import numpy as np
# Pytorch Modules
import torch
import torch.nn.functional as F

def check_path(path: str):
    """
    Check if the path exists and create it if not.
    """
    if not os.path.exists(path):
        os.makedirs(path)

def set_random_seed(seed: int):
    """
    Set random seed for reproducibility.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def get_torch_device(device: str):
    if device is not None:
        get_torch_device.device = device

    if 'cuda' in get_torch_device.device: # This also supports Rocm by amd gpu.
        if torch.cuda.is_available():
            return torch.device(get_torch_device.device) # This is for multi-gpu environment, e.g. 'cuda:0'
        else:
            print("No GPU found. Using CPU.")
            return torch.device('cpu')
    elif 'mps' in device: # This is for apple-silicon macs. requires pytorch 1.12+
        if not torch.backends.mps.is_available():
            if not torch.backends.mps.is_built():
                print("MPS not available because the current PyTorch install"
                      " was not built with MPS enabled.")
                print("Using CPU.")
            else:
                print("MPS not available because the current MacOS version"
                      " is not 12.3+ and/or you do not have an MPS-enabled"
                      " device on this machine.")
                print("Using CPU.")
            return torch.device('cpu')
        else:
            return torch.device(get_torch_device.device)
    elif 'cpu' in device:
        return torch.device('cpu')
    else:
        print("No such device found. Using CPU.")
        return torch.device('cpu')

class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.DEBUG):
        super().__init__(level)
        self.stream = sys.stdout

    def flush(self):
        self.acquire()
        try:
            if self.stream and hasattr(self.stream, "flush"):
                self.stream.flush()
        finally:
            self.release()

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg, self.stream)
            self.flush()
        except (KeyboardInterrupt, SystemExit, RecursionError):
            raise
        except Exception:
            self.handleError(record)

def write_log(logger, message):
    if logger:
        logger.info(message)

def get_tb_exp_name(args: argparse.Namespace):
    """
    Get the experiment name for tensorboard experiment.
    """

    ts = time.strftime('%Y-%b-%d-%H:%M:%S', time.localtime())

    exp_name = str()
    exp_name += "%s - " % args.task.upper()
    exp_name += "%s - " % args.proj_name

    if args.job in ['training', 'resume_training']:
        exp_name += 'TRAIN - '
        exp_name += "MODEL=%s - " % args.model_type.upper()
        exp_name += "DATA=%s - " % args.task_dataset.upper()
        exp_name += "AUG=%s_" % args.augmentation_type.upper()
        exp_name += "DESC=%s - " % args.description
    elif args.job == 'testing':
        exp_name += 'TEST - '
        exp_name += "MODEL=%s - " % args.model_type.upper()
        exp_name += "DATA=%s - " % args.task_dataset.upper()
        exp_name += "AUG=%s_" % args.augmentation_type.upper()
        exp_name += "DESC=%s - " % args.description
    exp_name += "TS=%s" % ts

    return exp_name

def get_huggingface_model_name(model_type: str) -> str:
    name = model_type.lower()

    if name in ['bert', 'cnn', 'lstm']: # 'cnn' and 'lstm' shares bert tokenizer.
        return 'bert-base-uncased'
    elif name == 'bart':
        return 'facebook/bart-large'
    elif name == 't5':
        return 't5-large'
    elif name == 'roberta':
        return 'roberta-base'
    else:
        raise NotImplementedError

def parse_bool(value: str):
    if value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
