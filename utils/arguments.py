# Standard Library Modules
import os
import argparse
# Custom Modules
from utils.utils import parse_bool

class ArgParser():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.user_name = os.getlogin()
        self.proj_name = 'SoftEDA'

        # Task arguments
        task_list = ['classification']
        self.parser.add_argument('--task', type=str, choices=task_list, default='classification',
                                 help='Task to do; Must be given.')
        job_list = ['preprocessing', 'training', 'resume_training', 'testing', 'augmentation']
        self.parser.add_argument('--job', type=str, choices=job_list, default='training',
                                 help='Job to do; Must be given.')
        dataset_list = ['imdb', 'sst2', 'cola', 'trec', 'subj', 'agnews', 'mr', 'cr', 'proscons']
        self.parser.add_argument('--task_dataset', type=str, choices=dataset_list, default='cola',
                                 help='Dataset for the task; Must be given.')
        self.parser.add_argument('--description', type=str, default='default',
                                 help='Description of the experiment; Default is "default"')

        # Path arguments
        self.parser.add_argument('--data_path', type=str, default='/HDD/dataset',
                                 help='Path to the raw dataset before preprocessing.')
        self.parser.add_argument('--preprocess_path', type=str, default=f'/HDD/{self.user_name}/preprocessed',
                                 help='Path to the preprocessed dataset.')
        self.parser.add_argument('--model_path', type=str, default=f'/HDD/{self.user_name}/model_final/{self.proj_name}',
                                 help='Path to the model after training.')
        self.parser.add_argument('--checkpoint_path', type=str, default=f'/HDD/{self.user_name}/model_checkpoint/{self.proj_name}')
        self.parser.add_argument('--result_path', type=str, default=f'/HDD/{self.user_name}/results/{self.proj_name}',
                                 help='Path to the result after testing.')
        self.parser.add_argument('--log_path', type=str, default=f'/HDD/{self.user_name}/tensorboard_log/{self.proj_name}',
                                 help='Path to the tensorboard log file.')

        # Model - Basic arguments
        self.parser.add_argument('--proj_name', type=str, default='SoftEDA',
                                 help='Name of the project.')
        model_type_list = ['bert', 'roberta', 'cnn', 'lstm']
        self.parser.add_argument('--model_type', type=str, choices=model_type_list, default='bert',
                                 help='Type of the classification model to use.')
        self.parser.add_argument('--model_ispretrained', type=parse_bool, default=True,
                                 help='Whether to use pretrained model; Default is True')
        self.parser.add_argument('--min_seq_len', type=int, default=4,
                                 help='Minimum sequence length of the input; Default is 4')
        self.parser.add_argument('--max_seq_len', type=int, default=100,
                                 help='Maximum sequence length of the input; Default is 100')
        self.parser.add_argument('--dropout_rate', type=float, default=0.2,
                                 help='Dropout rate of the model; Default is 0.2')

        # Model - Size arguments
        self.parser.add_argument('--embed_size', type=int, default=768, # Will be automatically specified by the model type if model is PLM
                                 help='Embedding size of the model; Default is 768')
        self.parser.add_argument('--hidden_size', type=int, default=768, # Will be automatically specified by the model type if model is PLM
                                 help='Hidden size of the model; Default is 768')

        # Model - Optimizer & Scheduler arguments
        optim_list = ['SGD', 'AdaDelta', 'Adam', 'AdamW']
        scheduler_list = ['None', 'StepLR', 'LambdaLR', 'CosineAnnealingLR', 'CosineAnnealingWarmRestarts', 'ReduceLROnPlateau']
        self.parser.add_argument('--optimizer', type=str, choices=optim_list, default='AdamW',
                                 help="Optimizer to use; Default is Adam")
        self.parser.add_argument('--scheduler', type=str, choices=scheduler_list, default='None',
                                 help="Scheduler to use for classification; If None, no scheduler is used; Default is None")

        # Training arguments 1
        self.parser.add_argument('--num_epochs', type=int, default=50,
                                 help='Training epochs; Default is 50')
        self.parser.add_argument('--learning_rate', type=float, default=1e-4,
                                 help='Learning rate of optimizer; Default is 1e-4')
        # Training arguments 2
        self.parser.add_argument('--num_workers', type=int, default=2,
                                 help='Num CPU Workers; Default is 2')
        self.parser.add_argument('--batch_size', type=int, default=32,
                                 help='Batch size; Default is 32')
        self.parser.add_argument('--weight_decay', type=float, default=1e-5,
                                 help='Weight decay; Default is 5e-4; If 0, no weight decay')
        self.parser.add_argument('--clip_grad_norm', type=int, default=5,
                                 help='Gradient clipping norm; Default is 5')
        self.parser.add_argument('--label_smoothing_eps', type=float, default=0.05,
                                 help='Label smoothing epsilon; Default is 0.05')
        self.parser.add_argument('--early_stopping_patience', type=int, default=5,
                                 help='Early stopping patience; No early stopping if None; Default is 5')
        self.parser.add_argument('--train_valid_split', type=float, default=0.2,
                                 help='Train/Valid split ratio; Default is 0.2')
        objective_list = ['loss', 'accuracy', 'f1']
        self.parser.add_argument('--optimize_objective', type=str, choices=objective_list, default='accuracy',
                                 help='Objective to optimize; Default is accuracy')

        # Augmentation arguments
        augmentation_type_list = ['none', 'hard_eda', 'aeda', 'soft_eda']
        self.parser.add_argument('--augmentation_type', type=str, choices=augmentation_type_list, default='none',
                                 help='Whether to use augmented data; Default is none')
        self.parser.add_argument('--augmentation_eda_alpha_sr', type=float, default=0.1,
                                 help='EDA alpha_sr; Default is 0.1')
        self.parser.add_argument('--augmentation_eda_alpha_ri', type=float, default=0.1,
                                 help='EDA alpha_ri; Default is 0.1')
        self.parser.add_argument('--augmentation_eda_alpha_rs', type=float, default=0.1,
                                 help='EDA alpha_rs; Default is 0.1')
        self.parser.add_argument('--augmentation_eda_p_rd', type=float, default=0.1,
                                 help='EDA p_rd; Default is 0.1')
        self.parser.add_argument('--augmentation_aeda_alpha', type=float, default=0.1,
                                 help='AEDA alpha; Default is 0.1')
        self.parser.add_argument('--augmentation_label_smoothing', type=float, default=0.2,
                                 help='Label smoothing value for SoftEDA; Default is 0.2')

        # Testing/Inference arguments
        self.parser.add_argument('--test_batch_size', default=16, type=int,
                                 help='Batch size for test; Default is 16')

        # Other arguments - Device, Seed, Logging, etc.
        self.parser.add_argument('--device', type=str, default='cuda:0',
                                 help='Device to use for training; Default is cuda')
        self.parser.add_argument('--seed', type=int, default=2023,
                                 help='Random seed; Default is 2023')
        self.parser.add_argument('--use_tensorboard', type=parse_bool, default=True,
                                 help='Using tensorboard; Default is True')
        self.parser.add_argument('--log_freq', default=500, type=int,
                                 help='Logging frequency; Default is 500')

    def get_args(self):
        return self.parser.parse_args()
