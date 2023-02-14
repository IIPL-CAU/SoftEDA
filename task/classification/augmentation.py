# Standard Library Modules
import os
import sys
import pickle
import argparse
# 3rd-party Modules
import bs4
import pandas as pd
from tqdm.auto import tqdm
# Pytorch Modules
import torch
# Huggingface Modules
from transformers import AutoTokenizer
# Custom Modules
from .augmentation_utils import run_eda, run_aeda
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils.utils import check_path, get_huggingface_model_name

def load_preprocessed_data(args: argparse.Namespace) -> dict:
    """
    Open preprocessed train pickle file from local directory.

    Args:
        args (argparse.Namespace): Arguments.

    Returns:
        train_data (dict): Preprocessed training data.
    """

    preprocessed_path = os.path.join(args.preprocess_path, args.task, args.task_dataset, args.model_type, 'train_processed.pkl')

    with open(preprocessed_path, 'rb') as f:
        train_data = pickle.load(f)

    return train_data

def augmentation(args: argparse.Namespace) -> None:
    """
    1. Load preprocessed train data
    2. Apply augmentation by pre-defined augmentation strategy & Give soft labels for soft_eda method
    3. Concatenate original data & augmented data
    4. Save total data

    Args:
        args (argparse.Namespace): Arguments.
    """

    # Load preprocessed train data
    train_data = load_preprocessed_data(args)
    augmented_data = {
        'input_ids': [],
        'attention_mask': [],
        'token_type_ids': [],
        'labels': [],
        'soft_labels': [],
        'num_classes': train_data['num_classes'],
        'vocab_size': train_data['vocab_size'],
        'pad_token_id': train_data['pad_token_id']
    }

    # Reconstruct sentence from input_ids using huggingface tokenizer
    model_name = get_huggingface_model_name(args.model_type)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    for idx in tqdm(range(len(train_data['input_ids'])), desc=f'Augmenting with {args.augmentation_type}'):
        decoded_sent = tokenizer.decode(train_data['input_ids'][idx], skip_special_tokens=True)

        # Apply augmentation by pre-defined augmentation strategy
        if args.augmentation_type == 'hard_eda':
            augmented_sent = run_eda(decoded_sent, args)
            augmented_data['soft_labels'].append(train_data['soft_labels'][idx]) # Hard EDA: Keep original soft labels (Actually one-hot labels)
        elif args.augmentation_type == 'soft_eda':
            augmented_sent = run_eda(decoded_sent, args)
            soft_labels = train_data['soft_labels'][idx] * (1 - args.augmentation_label_smoothing) + args.augmentation_label_smoothing / train_data['num_classes']
            augmented_data['soft_labels'].append(soft_labels) # SoftEDA: Apply soft labels for soft_eda method using label smoothing
        elif args.augmentation_type == 'aeda':
            augmented_sent = run_aeda(decoded_sent, args)
            augmented_data['soft_labels'].append(train_data['soft_labels'][idx]) # AEDA: Keep original soft labels (Actually one-hot labels)

        # Encode augmented sentence using huggingface tokenizer
        tokenized_sent = tokenizer(augmented_sent, padding='max_length', truncation=True,
                                   max_length=args.max_seq_len, return_tensors='pt')

        # Append augmented data
        augmented_data['input_ids'].append(tokenized_sent['input_ids'].squeeze())
        augmented_data['attention_mask'].append(tokenized_sent['attention_mask'].squeeze())
        if args.model_type == 'bert':
            augmented_data['token_type_ids'].append(tokenized_sent['token_type_ids'].squeeze())
        else:
            augmented_data['token_type_ids'].append(torch.zeros(args.max_seq_len, dtype=torch.long))
        augmented_data['labels'].append(train_data['labels'][idx]) # Keep original label

    # Merge original data & augmented data
    total_dict = {
        'input_ids': train_data['input_ids'] + augmented_data['input_ids'],
        'attention_mask': train_data['attention_mask'] + augmented_data['attention_mask'],
        'token_type_ids': train_data['token_type_ids'] + augmented_data['token_type_ids'],
        'labels': train_data['labels'] + augmented_data['labels'],
        'soft_labels': train_data['soft_labels'] + augmented_data['soft_labels'],
        'num_classes': train_data['num_classes'],
        'vocab_size': train_data['vocab_size'],
        'pad_token_id': train_data['pad_token_id']
    }

    # Save total data as pickle file
    save_path = os.path.join(args.preprocess_path, args.task, args.task_dataset, args.model_type)
    with open(os.path.join(save_path, f'train_augmented_{args.augmentation_type}.pkl'), 'wb') as f:
        pickle.dump(total_dict, f)
