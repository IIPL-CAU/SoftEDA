# Standard Library Modules
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import json
import argparse
# Custom Modules
from task.classification.preprocessing import preprocessing
from task.classification.augmentation import augmentation
from task.classification.train import training
from task.classification.test import testing
from utils.arguments import ArgParser
from utils.utils import check_path, set_random_seed

MODELS = ['bert', 'cnn', 'lstm', 'roberta']
DATASETS = ['trec', 'sst2', 'subj', 'agnews', 'mr', 'cr', 'proscons']
AUG_TYPES = ['none', 'hard_eda', 'soft_eda', 'aeda']
SOFT_EDA_SMOOTHINGS = [0.1, 0.15, 0.2, 0.25, 0.3]

def compose_result_dict(args: argparse.Namespace, acc: float, f1: float) -> dict:
    result_dict = {
        'dataset': args.task_dataset,
        'model': args.model_type,
        'aug_type': args.augmentation_type,
        'soft_eda_smoothing': args.augmentation_label_smoothing if args.augmentation_type == 'soft_eda' else 0.0,
        'test_acc': acc,
        'test_f1': f1,
    }
    return result_dict

def total_search(args: argparse.Namespace) -> None:
    result_list = []
    args.device = 'cuda:3'
    for each_dataset in DATASETS:
        args.task_dataset = each_dataset

        for each_model in MODELS:
            args.model_type = each_model
            preprocessing(args)

            for each_aug_type in AUG_TYPES:
                args.augmentation_type = each_aug_type
                if each_aug_type == 'none':
                    args.description = 'no_aug'
                    training(args)
                    acc, f1 = testing(args)
                    result_dict = compose_result_dict(args, acc, f1)
                elif each_aug_type == 'hard_eda':
                    args.description = 'alpha=0.1,aug=1'
                    augmentation(args)
                    training(args)
                    acc, f1 = testing(args)
                    result_dict = compose_result_dict(args, acc, f1)
                elif each_aug_type == 'aeda':
                    args.description = 'alpha=0.1,aug=1'
                    augmentation(args)
                    training(args)
                    acc, f1 = testing(args)
                    result_dict = compose_result_dict(args, acc, f1)
                elif each_aug_type == 'soft_eda':
                    for each_soft_eda_smoothing in SOFT_EDA_SMOOTHINGS:
                        args.augmentation_label_smoothing = each_soft_eda_smoothing
                        augmentation(args)
                        args.description = f'smoothing={each_soft_eda_smoothing},aug=1'
                        training(args)
                        acc, f1 = testing(args)
                        result_dict = compose_result_dict(args, acc, f1)
                else:
                    raise ValueError(f'Invalid augmentation type: {each_aug_type}')

                result_list.append(result_dict)

    # Save the result as a json file
    with open(os.path.join(args.result_path, 'total_search.json'), 'w') as f:
        json.dump(result_list, f, indent=4)

if __name__ == '__main__':
    # Parse arguments
    parser = ArgParser()
    args = parser.get_args()

    # Set random seed
    if args.seed is not None:
        set_random_seed(args.seed)

    # Check if the path exists
    for path in []:
        check_path(path)

    # Run the main function
    total_search(args)
