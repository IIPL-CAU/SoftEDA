# Standard Library Modules
import os
import sys
import logging
import argparse
# 3rd-party Modules
from tqdm.auto import tqdm
from sklearn.metrics import f1_score
# Pytorch Modules
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
# Custom Modules
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from model.classification.model import ClassificationModel
from model.classification.dataset import CustomDataset
from utils.utils import TqdmLoggingHandler, write_log, get_tb_exp_name, get_torch_device

def testing(args: argparse.Namespace) -> tuple: # (test_acc_cls, test_f1_cls)
    device = get_torch_device(args.device)

    # Define logger and tensorboard writer
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    handler = TqdmLoggingHandler()
    handler.setFormatter(logging.Formatter(" %(asctime)s - %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(handler)
    logger.propagate = False

    if args.use_tensorboard:
        writer = SummaryWriter(os.path.join(args.log_path, get_tb_exp_name(args)))
        writer.add_text('args', str(args))

    # Load dataset and define dataloader
    write_log(logger, "Loading dataset")
    dataset_test = CustomDataset(os.path.join(args.preprocess_path, args.task, args.task_dataset, args.model_type, f'test_processed.pkl'))
    dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size, num_workers=args.num_workers,
                                 shuffle=False, pin_memory=True, drop_last=False)
    args.vocab_size = dataset_test.vocab_size
    args.num_classes = dataset_test.num_classes
    args.pad_token_id = dataset_test.pad_token_id

    write_log(logger, "Loaded data successfully")
    write_log(logger, f"Test dataset size / iterations: {len(dataset_test)} / {len(dataloader_test)}")

    # Get model instance
    write_log(logger, "Building model")
    model = ClassificationModel(args).to(device)

    # Load model weights
    write_log(logger, "Loading model weights")
    load_model_name = os.path.join(args.model_path, args.task, args.task_dataset, args.model_type, 'final_model.pt')
    model = model.to('cpu')
    checkpoint = torch.load(load_model_name, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    write_log(logger, f"Loaded model weights from {load_model_name}")
    del checkpoint

    # Define loss function
    cls_loss = nn.CrossEntropyLoss()
    write_log(logger, f"Loss function: {cls_loss}")

    # Test - Start testing
    model = model.eval()
    test_loss_cls = 0
    test_acc_cls = 0
    test_f1_cls = 0
    for test_iter_idx, data_dicts in enumerate(tqdm(dataloader_test, total=len(dataloader_test), desc="Testing")):
        # Test - Get input data
        input_ids = data_dicts['input_ids'].to(device)
        attention_mask = data_dicts['attention_mask'].to(device)
        token_type_ids = data_dicts['token_type_ids'].to(device)
        soft_labels = data_dicts['soft_labels'].to(device)
        hard_labels = data_dicts['labels'].to(device) # For calculating accuracy

        # Test - Forward pass
        with torch.no_grad():
            classification_logits = model(input_ids, attention_mask, token_type_ids)

        # Test - Calculate loss & acc/f1 score
        batch_loss_cls = cls_loss(classification_logits, soft_labels)
        batch_acc_cls = (classification_logits.argmax(dim=-1) == hard_labels).float().mean()
        batch_f1_cls = f1_score(hard_labels.cpu().numpy(), classification_logits.argmax(dim=-1).cpu().numpy(), average='macro')

        # Test - Logging
        test_loss_cls += batch_loss_cls.item()
        test_acc_cls += batch_acc_cls.item()
        test_f1_cls += batch_f1_cls

        if test_iter_idx % args.log_freq == 0 or test_iter_idx == len(dataloader_test) - 1:
            write_log(logger, f"TEST - Iter [{test_iter_idx}/{len(dataloader_test)}] - Loss: {batch_loss_cls.item():.4f}")
            write_log(logger, f"TEST - Iter [{test_iter_idx}/{len(dataloader_test)}] - Acc: {batch_acc_cls.item():.4f}")
            write_log(logger, f"TEST - Iter [{test_iter_idx}/{len(dataloader_test)}] - F1: {batch_f1_cls:.4f}")

    # Test - Check loss
    test_loss_cls /= len(dataloader_test)
    test_acc_cls /= len(dataloader_test)
    test_f1_cls /= len(dataloader_test)

    # Final - End of testing
    write_log(logger, f"Done! - TEST - Loss: {test_loss_cls:.4f} - Acc: {test_acc_cls:.4f} - F1: {test_f1_cls:.4f}")
    if args.use_tensorboard:
        writer.add_scalar('TEST/Loss', test_loss_cls, 0)
        writer.add_scalar('TEST/Acc', test_acc_cls, 0)
        writer.add_scalar('TEST/F1', test_f1_cls, 0)
        writer.close()

    return test_acc_cls, test_f1_cls
