"""
This module handles model training, validation, and fold-wise setup for the ArgueFlow pipeline.
"""

# import logging
import os
import random

# import time
import warnings

import numpy as np

# import pandas as pd
import torch


# from imp import reload


# For SWA
# from transformers import (
#     AutoConfig,
#     AutoTokenizer,
# )


warnings.filterwarnings('ignore')

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

# """# Initial configuration"""


# class Config:
#     # General settings
#     competition_name = 'FeedbackPrize2'
#     env = 'local'
#     seed = 1
#     mode = 'train'  # 'train', 'valid'
#     debug = False
#     model_name = 'v1a'
#     processed_data = (
#         True  # Set to True if the train_df is processed and added with the target (rank)
#     )
#     use_tqdm = True
#     # For model
#     backbone = 'microsoft/deberta-v3-large'
#     tokenizer = AutoTokenizer.from_pretrained(backbone)
#     config = AutoConfig.from_pretrained(backbone)
#     config.output_hidden_states = True
#     config.hidden_dropout_prob = 0.0
#     config.attention_probs_dropout_prob = 0.0
#     # Add new token
#     cls_token = '[FP2]'
#     special_tokens_dict = {'additional_special_tokens': [cls_token]}
#     tokenizer.add_special_tokens(special_tokens_dict)
#     cls_token_id = tokenizer(cls_token)['input_ids'][1]
#     # For data
#     discourse_type_map = {
#         'Lead': 0,
#         'Position': 1,
#         'Claim': 2,
#         'Counterclaim': 3,
#         'Rebuttal': 4,
#         'Evidence': 5,
#         'Concluding Statement': 6,
#     }
#     label_map = {
#         'Ineffective': 0,
#         'Adequate': 1,
#         'Effective': 2,
#         'None': -100,
#     }
#     training_folds = [0]
#     max_len = 1024
#     batch_size = 4
#     num_workers = os.cpu_count()
#     # For training
#     apex = True
#     gradient_checkpointing = True
#     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#     nepochs = 5
#     start_val_epoch = 2
#     val_check_interval = 0.1
#     gradient_accumulation_steps = 1.0
#     max_grad_norm = 1000
#     label_smoothing = 0.03
#     # For SWA
#     use_swa = False
#     if use_swa:
#         start_swa_epoch = 2
#         swa_lr = 5e-6
#         swa_anneal_strategy = 'cos'
#         swa_anneal_epochs = 1
#     else:
#         start_swa_epoch = nepochs + 1
#     # For AWP
#     use_awp = True
#     if use_awp:
#         start_awp_epoch = 2
#         adv_lr = 1e-5
#         adv_eps = 1e-3
#         adv_step = 1
#     else:
#         start_awp_epoch = nepochs + 1
#     # Optimizer
#     lr = 1e-5
#     weight_decay = 1e-2
#     encoder_lr = 1e-5
#     decoder_lr = 1e-5
#     min_lr = 1e-6
#     eps = 1e-6
#     betas = (0.9, 0.999)
#     # Scheduler
#     scheduler_type = 'cosine'  # 'linear', 'cosine'
#     if scheduler_type == 'cosine':
#         num_cycles = 0.5
#     num_warmup_steps = 100
#     batch_scheduler = True
#     # Directories
#     if env == 'colab':
#         comp_data_dir = (
#             f'/content/drive/My Drive/Kaggle competitions/{competition_name}/comp_data'
#         )
#         extra_data_dir = (
#             f'/content/drive/My Drive/Kaggle competitions/{competition_name}/extra_data'
#         )
#         model_dir = (
#             f'/content/drive/My Drive/Kaggle competitions/{competition_name}/model'
#         )
#         os.makedirs(
#             os.path.join(
#                 model_dir, model_name.split('_')[0][:-1], model_name.split('_')[0][-1]
#             ),
#             exist_ok=True,
#         )
#         old_comp_data_dir = (
#             f'/content/drive/My Drive/Kaggle competitions/{competition_name[:-1]}/data'
#         )
#     elif env == 'kaggle':
#         comp_data_dir = ...
#         extra_data_dir = ...
#         model_dir = ...
#     elif env == 'vastai':
#         comp_data_dir = 'data'
#         extra_data_dir = 'data'
#         model_dir = 'model'
#         os.makedirs(
#             os.path.join(
#                 model_dir, model_name.split('_')[0][:-1], model_name.split('_')[0][-1]
#             ),
#             exist_ok=True,
#         )
#     elif env == 'local':
#         comp_data_dir = 'data'
#         extra_data_dir = 'ext_data'
#         model_dir = 'model'
#         os.makedirs(
#             os.path.join(
#                 model_dir, model_name.split('_')[0][:-1], model_name.split('_')[0][-1]
#             ),
#             exist_ok=True,
#         )


# cfg = Config()

# """# Random seed"""


def set_random_seed(seed, use_cuda=True):
    """
    Sets random seeds for reproducibility across NumPy, PyTorch, and Python.

    Args:
        seed (int): The seed value to use.
        use_cuda (bool): Whether to apply CUDA-specific seed settings. Defaults to True.
    """
    np.random.seed(seed)  # cpu vars
    torch.manual_seed(seed)  # cpu  vars
    random.seed(seed)  # Python
    os.environ['PYTHONHASHSEED'] = str(seed)  # Python hash building
    if use_cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # gpu vars
        torch.backends.cudnn.deterministic = True  # needed
        torch.backends.cudnn.benchmark = False


# set_random_seed(cfg.seed)

# """# Set-up log"""


# # reload(logging)
# # logging.basicConfig(
# #     level=logging.INFO,
# #     format='%(asctime)s %(message)s',
# #     datefmt='%H:%M:%S',
# #     handlers=[
# #         logging.FileHandler(
# #             f"train_{cfg.model_name}_{time.strftime('%m%d_%H%M', time.localtime())}_{cfg.seed}.log"
# #         ),
# #         logging.StreamHandler(),
# #     ],
# # )

# logging.info(
#     '\nmodel_name: {}\n'
#     'env: {}\n'
#     'seed: {}\n'
#     'training_folds: {}\n'
#     'max_len: {}\n'
#     'batch_size: {}\n'
#     'num_workers: {}\n'
#     'nepochs: {}\n'
#     'lr: {}\n'
#     'weight_decay: {}\n'
#     'gradient_accumulation_steps: {}'.format(
#         cfg.model_name,
#         cfg.env,
#         cfg.seed,
#         cfg.training_folds,
#         cfg.max_len,
#         cfg.batch_size,
#         cfg.num_workers,
#         cfg.nepochs,
#         cfg.lr,
#         cfg.weight_decay,
#         cfg.gradient_accumulation_steps,
#     )
# )

# """# Import data

# * Train data
# """

# if cfg.processed_data:
#     train = pd.read_csv(os.path.join(cfg.extra_data_dir, f'train_seed_{cfg.seed}.csv'))
# else:
#     train = pd.read_csv(os.path.join(cfg.comp_data_dir, 'train.csv'))
# train
