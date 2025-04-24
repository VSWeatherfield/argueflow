"""
This module provides inference functions for the ArgueFlow pipeline,
including utilities to load models, run predictions, and organize outputs.
"""

# import gc
# import os
# import random
# import shutil
# import warnings
# from glob import glob
# from pathlib import Path
# from typing import List

# import numpy as np
# import pandas as pd
# import torch
# import torch.nn.functional as F
# import transformers
# from datasets import Dataset
# from IPython.display import display
# from scipy.special import softmax
# from sklearn.metrics import log_loss
# from torch import nn
# from torch.cuda.amp import autocast
# from torch.optim.swa_utils import AveragedModel
# from torch.utils.data import DataLoader
# from tqdm.notebook import tqdm
# from transformers import AutoConfig, AutoModel
# from transformers.models.deberta_v2.tokenization_deberta_v2_fast import (
#     DebertaV2TokenizerFast,
# )


# transformers_path = Path('/opt/conda/lib/python3.7/site-packages/transformers')

# input_dir = Path('../input/deberta-v2-3-fast-tokenizer')

# convert_file = input_dir / 'convert_slow_tokenizer.py'
# conversion_path = transformers_path / convert_file.name

# if conversion_path.exists():
#     conversion_path.unlink()

# shutil.copy(convert_file, transformers_path)
# deberta_v2_path = transformers_path / 'models' / 'deberta_v2'

# for filename in [
#     'tokenization_deberta_v2.py',
#     'tokenization_deberta_v2_fast.py',
#     'deberta__init__.py',
# ]:
#     if str(filename).startswith('deberta'):
#         filepath = deberta_v2_path / str(filename).replace('deberta', '')
#     else:
#         filepath = deberta_v2_path / filename
#     if filepath.exists():
#         filepath.unlink()

#     shutil.copy(input_dir / filename, filepath)


# warnings.filterwarnings('ignore')

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# os.environ['TOKENIZERS_PARALLELISM'] = 'true'


# transformers.__version__


# class Config:
#     # General settings
#     competition_name = 'FeedbackPrize2'
#     env = 'kaggle'
#     mode = 'infer'  # 'infer', 'valid', 'infer_pl'
#     seed = 1
#     debug = False
#     model_name = 'v12h'
#     use_tqdm = True
#     # For model
#     if env == 'colab':
#         backbone = 'microsoft/deberta-v3-large'
#     elif env == 'kaggle':
#         backbone = '../input/microsoftdebertav3large'
#     tokenizer = DebertaV2TokenizerFast.from_pretrained(backbone)
#     config = AutoConfig.from_pretrained(backbone)
#     config.output_hidden_states = True
#     config.hidden_dropout_prob = 0.0
#     config.attention_probs_dropout_prob = 0.0
#     # Add new token
#     cls_token = '[FP2]'
#     cls_token_id = tokenizer.vocab_size + 1
#     special_tokens_dict = {'additional_special_tokens': [cls_token]}
#     tokenizer.add_special_tokens(special_tokens_dict)
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
#     training_folds = [0, 1, 2, 3, 4]
#     max_len = 1024
#     batch_size = 16
#     num_workers = os.cpu_count()
#     # For training
#     apex = False
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
#     elif env == 'kaggle':
#         comp_data_dir = '../input/feedback-prize-effectiveness'
#         extra_data_dir = '../input/feedbackprize2extradata'
#         model_dir = f'../input/feedbackpriz2{model_name}'


# cfg = Config()


# def set_random_seed(seed, use_cuda=True):
#     np.random.seed(seed)  # cpu vars
#     torch.manual_seed(seed)  # cpu  vars
#     random.seed(seed)  # Python
#     os.environ['PYTHONHASHSEED'] = str(seed)  # Python hash building
#     if use_cuda:
#         torch.cuda.manual_seed(seed)
#         torch.cuda.manual_seed_all(seed)  # gpu vars
#         torch.backends.cudnn.deterministic = True  # needed
#         torch.backends.cudnn.benchmark = False


# if cfg.mode == 'infer':
#     test = pd.read_csv(os.path.join(cfg.comp_data_dir, 'test.csv'))
#     test['discourse_effectiveness'] = 'None'
#     test['fold'] = -1
#     test[list(cfg.label_map.keys())[:3]] = -100
# elif cfg.mode == 'valid':
#     test = pd.read_csv(os.path.join(cfg.extra_data_dir, 'train_seed_1.csv'))
#     one_hot_label = np.zeros((test.shape[0], 3))
#     one_hot_label[
#         (
#             np.arange(test.shape[0]),
#             test['discourse_effectiveness'].map(cfg.label_map).values,
#         )
#     ] = 1
#     test[list(cfg.label_map.keys())[:3]] = one_hot_label
# elif cfg.mode == 'infer_pl':
#     test = pd.read_csv(os.path.join(cfg.extra_data_dir, 'test_pl.csv'))
#     test['discourse_id'] = (
#         test['discourse_id'].astype(str) + '_' + test['fold'].astype(str)
#     )
#     test['essay_id'] = test['essay_id'].astype(str) + '_' + test['fold'].astype(str)

# discourse_map = dict(zip(test['discourse_id'].astype(str), test.index, strict=True))
# inv_discourse_map = {v: k for k, v in discourse_map.items()}
# test


# # Read each essay
# if cfg.mode == 'infer':
#     essay_paths = glob(f'{cfg.comp_data_dir}/test/*.txt')
#     essay_dict = {
#         'essay_id': [i.split('/')[-1].split('.')[0] for i in tqdm(essay_paths)],
#         'text': [open(i).read() for i in tqdm(essay_paths)],
#     }
#     essay_df = pd.DataFrame.from_dict(essay_dict)
#     assert essay_df.shape[0] == test.essay_id.nunique()
#     display(essay_df)


# if cfg.mode == 'infer':
#     test = test.merge(essay_df, right_on='essay_id', left_on='essay_id')
# test


# def organize_df(df, idx, cfg):
#     """
#     Converts a DataFrame of discourse units into a single-row summary.

#     Args:
#         df (pd.DataFrame): Sample-level data with columns like 'discourse_id', 'discourse_type',
#             'discourse_text', and label columns from `cfg.label_map`.
#         idx: Index to assign to the output row.
#         cfg: Config object with `cls_token` and `label_map`.

#     Returns:
#         pd.DataFrame: Single-row DataFrame with merged text, labels, and metadata.
#     """
#     discourse_ids = '|'.join(df['discourse_id'].astype(str).tolist())
#     discourses = ''.join(
#         (cfg.cls_token + df['discourse_type'] + '. ' + df['discourse_text']).tolist()
#     )
#     label_list = '|'.join(
#         df[list(cfg.label_map.keys())[:3]].values.reshape(-1).astype(str).tolist()
#     )
#     label_name_list = '|'.join(df['discourse_effectiveness'].tolist())
#     fold = df['fold'].unique()[0]
#     return pd.DataFrame(
#         {
#             'discourse_ids': discourse_ids,
#             'discourses': discourses,
#             'label_list': label_list,
#             'label_name_list': label_name_list,
#             'fold': fold,
#         },
#         index=[idx],
#     )


# test_df = []
# for idx, df in tqdm(test.groupby('essay_id')):
#     test_df.append(organize_df(df, idx, cfg))
# test_df = pd.concat(test_df).reset_index()
# test.set_index('discourse_id', inplace=True)
# test_df


# def create_label(
#     cfg, input_ids, overflow_to_sample_mapping, label_list, discourse_id_list
# ):
#     discourse_ids = []
#     labels = []
#     is_tail = []
#     for _, (input_id, sample_mapping) in enumerate(
#         zip(input_ids, overflow_to_sample_mapping, strict=True)
#     ):
#         input_ids_array = np.array(input_id)
#         label = np.zeros((len(input_id), 3)) - 100
#         discourse_id = np.zeros_like(input_id) - 100
#         current_label = label_list[sample_mapping]
#         # Recover the label
#         current_label = np.array(current_label.split('|')).astype(float).reshape(-1, 3)
#         num_cls_tokens = sum(input_ids_array == cfg.cls_token_id)
#         start = 0
#         is_tail.append(0)

#         end = start + num_cls_tokens

#         chosen_tokens = input_ids_array == cfg.cls_token_id
#         label[chosen_tokens] = current_label[start:end]
#         discourse_id[chosen_tokens] = discourse_id_list[sample_mapping][start:end]
#         labels.append(label)
#         discourse_ids.append(discourse_id)

#     return labels, discourse_ids, is_tail


# def prepare_dataset(example, cfg=cfg, mode='train'):
#     discourses = example['discourses']
#     label_list = example['label_list']
#     discourse_id_list = [
#         [discourse_map[i] for i in lala.split('|')] for lala in example['discourse_ids']
#     ]

#     if mode == 'train' or mode == 'valid':
#         padding = 'max_length'
#     else:
#         padding = False

#     tokenized_discourses = cfg.tokenizer(
#         discourses,
#         return_attention_mask=True,
#         truncation=True,
#         max_length=cfg.max_len,
#         padding=padding,
#         return_length=True,
#         return_overflowing_tokens=True,
#     )
#     input_ids = tokenized_discourses['input_ids']
#     overflow_to_sample_mapping = tokenized_discourses['overflow_to_sample_mapping']
#     assert max(overflow_to_sample_mapping) == len(label_list) - 1

#     labels, discourse_ids, is_tail = create_label(
#         cfg, input_ids, overflow_to_sample_mapping, label_list, discourse_id_list
#     )
#     tokenized_discourses['labels'] = labels
#     tokenized_discourses['discourse_ids'] = discourse_ids
#     tokenized_discourses['is_tail'] = is_tail

#     return tokenized_discourses


# def generate_dataset(cfg, df, mode='train'):
#     ds = Dataset.from_pandas(df)
#     tokenized_ds = ds.map(
#         prepare_dataset,
#         fn_kwargs={'cfg': cfg, 'mode': mode},
#         batched=True,
#         batch_size=20_000,
#         remove_columns=ds.column_names,
#     )
#     if mode == 'infer':
#         # Sort dataset
#         len_sortidx = np.argsort(tokenized_ds['length'])
#         len_rank = np.vectorize(lambda x: len_sortidx.tolist().index(x))(
#             np.arange(len(len_sortidx))
#         )
#         sorted_tokenized_ds = tokenized_ds.sort('length')
#     else:
#         sorted_tokenized_ds, len_rank = None, None
#     return tokenized_ds, sorted_tokenized_ds, len_rank


# class FeedbackPrize2_Dataset(Dataset):
#     def __init__(self, cfg, df, mode='train'):
#         dataset, sorted_dataset, len_rank = generate_dataset(cfg, df, mode=mode)
#         self.dataset = dataset
#         self.mode = mode
#         if mode == 'infer':
#             self.dataset = sorted_dataset
#             self.len_rank = len_rank

#     def __len__(self):
#         return len(self.dataset)

#     def __getitem__(self, idx):
#         item = self.dataset[idx]
#         if self.mode == 'infer':
#             item['len_rank'] = self.len_rank[idx]
#         else:
#             item['len_rank'] = None
#         return item


# def pad_sequence(sequences: List[List[int]], max_len, padding_side='right', dim=1):
#     if padding_side == 'right':
#         if dim > 1:
#             return [x + [[-100] * dim] * (max_len - len(x)) for x in sequences]
#         else:
#             return [x + [-100] * (max_len - len(x)) for x in sequences]
#     else:
#         if dim > 1:
#             return [[[-100] * dim] * (max_len - len(x)) + x for x in sequences]
#         else:
#             return [[-100] * (max_len - len(x)) + x for x in sequences]


# def collate_fn(batch):
#     input_ids = []
#     attention_mask = []
#     token_type_ids = []
#     is_tail = []
#     labels = []
#     discourse_ids = []
#     len_rank = []

#     max_len = max([len(item['input_ids']) for item in batch])

#     for item in batch:
#         batch_input_ids = item['input_ids']
#         batch_attention_mask = item['attention_mask']
#         batch_token_type_ids = item['token_type_ids']
#         batch_is_tail = item['is_tail']
#         batch_labels = item['labels']
#         batch_discourse_ids = item['discourse_ids']

#         input_ids.append(batch_input_ids)
#         attention_mask.append(batch_attention_mask)
#         token_type_ids.append(batch_token_type_ids)
#         is_tail.append(batch_is_tail)
#         labels.append(batch_labels)
#         discourse_ids.append(batch_discourse_ids)

#         if item['len_rank'] is not None:
#             batch_len_rank = item['len_rank']
#             len_rank.append(batch_len_rank)
#         else:
#             len_rank.append(-1)

#     input_dict = {
#         'input_ids': input_ids,
#         'attention_mask': attention_mask,
#         'token_type_ids': token_type_ids,
#     }

#     # Pad the inputs
#     input_dict = cfg.tokenizer.pad(input_dict)
#     labels = pad_sequence(labels, max_len=max_len, dim=3)
#     discourse_ids = pad_sequence(discourse_ids, max_len=max_len)

#     return {
#         'input_ids': torch.tensor(input_dict['input_ids'], dtype=torch.long),
#         'attention_mask': torch.tensor(input_dict['attention_mask'], dtype=torch.long),
#         'token_type_ids': torch.tensor(input_dict['token_type_ids'], dtype=torch.long),
#         'is_tail': torch.tensor(is_tail, dtype=torch.long),
#         'labels': torch.tensor(labels, dtype=torch.float),
#         'discourse_ids': torch.tensor(discourse_ids, dtype=torch.long),
#         'len_rank': torch.tensor(len_rank, dtype=torch.long),
#     }


# dataset = FeedbackPrize2_Dataset(cfg, test_df, mode='infer')
# dataloader = DataLoader(
#     dataset, batch_size=4, num_workers=2, shuffle=False, collate_fn=collate_fn
# )
# item = next(iter(dataloader))
# item


# class FeedbackPrize2_Model(nn.Module):
#     def __init__(self, cfg):
#         super(FeedbackPrize2_Model, self).__init__()
#         self.cfg = cfg
#         # Backbone
#         self.backbone = AutoModel.from_pretrained(
#             cfg.backbone, config=cfg.config, ignore_mismatched_sizes=True
#         )
#         self.backbone.resize_token_embeddings(len(cfg.tokenizer))

#         # Multidropout
#         self.dropout1 = nn.Dropout(0.1)
#         self.dropout2 = nn.Dropout(0.2)
#         self.dropout3 = nn.Dropout(0.3)
#         self.dropout4 = nn.Dropout(0.4)
#         self.dropout5 = nn.Dropout(0.5)

#         # GRU
#         self.rnn = nn.GRU(
#             input_size=cfg.config.hidden_size,
#             hidden_size=cfg.config.hidden_size // 2,
#             bidirectional=True,
#             batch_first=True,
#             dropout=0.1,
#             num_layers=1,
#         )

#         # Head
#         self.head = nn.Linear(cfg.config.hidden_size, 3)

#         self._init_weights(self.head)

#     def _init_weights(self, module):
#         if isinstance(module, nn.Linear):
#             module.weight.data.normal_(mean=0.0, std=cfg.config.initializer_range)
#             if module.bias is not None:
#                 module.bias.data.zero_()
#         elif isinstance(module, nn.Embedding):
#             module.weight.data.normal_(mean=0.0, std=cfg.config.initializer_range)
#             if module.padding_idx is not None:
#                 module.weight.data[module.padding_idx].zero_()
#         elif isinstance(module, nn.LayerNorm):
#             module.bias.data.zero_()
#             module.weight.data.fill_(1.0)

#     def criterion(self, pred, true):
#         loss = nn.CrossEntropyLoss(ignore_index=-100)(pred.permute(0, 2, 1), true)
#         return loss

#     def forward(self, input_ids, attention_mask, token_type_ids, label=None):
#         output_backbone = self.backbone(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids,
#         ).last_hidden_state  # batch_size, seq_len, hidden_size
#         output_backbone = self.rnn(output_backbone)[0]

#         output1 = self.head(self.dropout1(output_backbone))
#         output2 = self.head(self.dropout2(output_backbone))
#         output3 = self.head(self.dropout3(output_backbone))
#         output4 = self.head(self.dropout4(output_backbone))
#         output5 = self.head(self.dropout5(output_backbone))

#         output = (output1 + output2 + output3 + output4 + output5) / 5

#         if label is not None:
#             loss = self.criterion(output, label)
#         else:
#             loss = None
#         _PADDING_VALUE = -1e30 if output.dtype == torch.float32 else -1e4
#         return loss, F.pad(
#             output,
#             (0, 0, 0, self.cfg.max_len - output.shape[1]),
#             'constant',
#             _PADDING_VALUE,
#         )


# def metric(y_pred, y_true):
#     return log_loss(y_true, y_pred)


# def infer_fn(cfg, model, infer_dataloader):
#     # Set up for training
#     model.eval()

#     preds = []
#     all_input_ids = []
#     all_is_tails = []
#     all_len_rank = []
#     all_discourse_ids = []

#     if cfg.use_tqdm:
#         tbar = tqdm(infer_dataloader)
#     else:
#         tbar = infer_dataloader

#     for _, item in enumerate(tbar):
#         # Set up inputs
#         input_ids = item['input_ids'].to(cfg.device)
#         attention_mask = item['attention_mask'].to(cfg.device)
#         token_type_ids = item['token_type_ids'].to(cfg.device)
#         is_tail = item['is_tail']
#         len_rank = item['len_rank']
#         discourse_ids = item['discourse_ids']

#         # Forward
#         with torch.no_grad():
#             with autocast(enabled=cfg.apex):
#                 _, batch_pred = model(input_ids, attention_mask, token_type_ids)

#         # Pad also the input_ids
#         input_ids = F.pad(
#             input_ids,
#             (0, cfg.max_len - input_ids.shape[1]),
#             'constant',
#             cfg.tokenizer.pad_token_id,
#         )
#         discourse_ids = F.pad(
#             discourse_ids, (0, cfg.max_len - discourse_ids.shape[1]), 'constant', -100
#         )

#         # Store the predictions
#         preds.append(batch_pred.detach().cpu().numpy())
#         all_input_ids.append(input_ids.detach().cpu().numpy())
#         all_is_tails.append(is_tail.numpy())
#         all_len_rank.append(len_rank.numpy())
#         all_discourse_ids.append(discourse_ids.numpy())

#     # Concatenate batch materials
#     preds = np.concatenate(preds, axis=0)
#     all_input_ids = np.concatenate(all_input_ids, axis=0)
#     all_is_tails = np.concatenate(all_is_tails, axis=0)
#     all_len_rank = np.concatenate(all_len_rank, axis=0)
#     all_discourse_ids = np.concatenate(all_discourse_ids, axis=0)

#     # Re-order the sorted predictions - size: (num of essays, max length, 3)
#     preds = preds[all_len_rank]
#     all_input_ids = all_input_ids[all_len_rank]
#     all_is_tails = all_is_tails[all_len_rank]
#     all_discourse_ids = all_discourse_ids[all_len_rank]

#     # Get predictions at the CLS tokens only, going from essays to discourses
#     preds = preds[np.where(all_input_ids == cfg.cls_token_id)]
#     discourse_ids = all_discourse_ids[np.where(all_input_ids == cfg.cls_token_id)]

#     return preds.astype(np.float64), np.array(
#         [inv_discourse_map[i] for i in discourse_ids]
#     )


# def inferring_loop(cfg, infer_dataloader, old_df=None, new_df=None, fold=0):
#     print(f' Fold {fold} '.center(50, '*'))
#     set_random_seed(cfg.seed + fold)

#     model = FeedbackPrize2_Model(cfg).to(cfg.device)
#     if cfg.env == 'colab':
#         model_path = os.path.join(
#             cfg.model_dir,
#             cfg.model_name.split('_')[0][:-1],
#             cfg.model_name.split('_')[0][-1],
#             f'fold_{fold}.pt',
#         )
#     elif cfg.env == 'kaggle':
#         model_path = os.path.join(cfg.model_dir, f'fold_{fold}.pt')
#     print(f'Loading the pre-trained model from {model_path}...')
#     ckp = torch.load(model_path, map_location=cfg.device)
#     try:
#         model.load_state_dict(ckp['state_dict'])
#         pred, discourse_ids = infer_fn(cfg, model, infer_dataloader)
#     except RuntimeError:
#         swa_model = AveragedModel(model)
#         swa_model.load_state_dict(ckp['state_dict'])
#         pred, discourse_ids = infer_fn(cfg, swa_model, infer_dataloader)

#     if cfg.mode == 'valid':
#         assert (old_df is not None) and (
#             new_df is not None
#         ), "The arguments 'old_df' and 'new_df' must be given in the validation mode!"
#         true = np.array(
#             [
#                 cfg.label_map[item]
#                 for sublist in [i.split('|') for i in new_df['label_name_list'].tolist()]
#                 for item in sublist
#             ]
#         )
#         discourse_ids = np.array(
#             [
#                 item
#                 for sublist in [i.split('|') for i in new_df['discourse_ids'].tolist()]
#                 for item in sublist
#             ]
#         )
#         old_df.loc[discourse_ids, list(cfg.label_map.keys())[:3]] = softmax(pred, axis=-1)

#         # Scoring
#         best_score = metric(softmax(pred, axis=-1), true)
#         print(f'Score: {best_score}')
#     elif cfg.mode == 'infer_pl':
#         discourse_ids = np.array(
#             [
#                 item
#                 for sublist in [i.split('|') for i in new_df['discourse_ids'].tolist()]
#                 for item in sublist
#             ]
#         )
#         old_df.loc[discourse_ids, list(cfg.label_map.keys())[:3]] = softmax(pred, axis=-1)

#     del model, ckp
#     torch.cuda.empty_cache()
#     gc.collect()

#     return pred, old_df, discourse_ids


# def main():
#     if cfg.mode == 'infer':
#         print('Preparing the inferring dataloader...')
#         test_dataset = FeedbackPrize2_Dataset(cfg, test_df, mode='infer')
#         test_dataloader = DataLoader(
#             test_dataset,
#             batch_size=cfg.batch_size,
#             num_workers=cfg.num_workers,
#             shuffle=False,
#             collate_fn=collate_fn,
#         )

#     oofs = []
#     for i, fold in enumerate(cfg.training_folds):
#         if cfg.mode == 'valid' or cfg.mode == 'infer_pl':
#             print(f'Preparing the dataloader for fold {fold}...')
#             val = test[test.fold == fold]
#             val_df = test_df[test_df.fold == fold]

#             test_dataset = FeedbackPrize2_Dataset(cfg, val_df, mode='infer')
#             test_dataloader = DataLoader(
#                 test_dataset,
#                 batch_size=cfg.batch_size,
#                 num_workers=cfg.num_workers,
#                 shuffle=False,
#                 collate_fn=collate_fn,
#             )

#         else:
#             val = None
#             val_df = None

#         fold_pred, val, discourse_ids = inferring_loop(
#             cfg, test_dataloader, old_df=val, new_df=val_df, fold=fold
#         )

#         if cfg.mode == 'infer':
#             if i == 0:
#                 pred = fold_pred / len(cfg.training_folds)
#             else:
#                 pred += fold_pred / len(cfg.training_folds)

#         if cfg.mode == 'infer_pl':
#             val.rename(columns={'index': 'discourse_id'}).to_csv(
#                 os.path.join(cfg.extra_data_dir, f'train_pl_{fold}.csv'), index=False
#             )

#         oofs.append(val)

#     if cfg.mode == 'valid':
#         oofs = pd.concat(oofs).loc[test.index]
#         pred = oofs[list(cfg.label_map.keys())[:3]].values
#         true = oofs['discourse_effectiveness'].map(cfg.label_map).values
#         # Scoring
#         score = metric(pred, true)
#         print('*' * 50)
#         print(f'OOF score: {score}')
#         oofs = oofs.reset_index()
#         display(oofs.head())
#         oofs.rename(columns={'index': 'discourse_id'}).to_pickle(
#             os.path.join(
#                 cfg.model_dir,
#                 cfg.model_name.split('_')[0][:-1],
#                 cfg.model_name.split('_')[0][-1],
#                 'oof_01234.pkl',
#             )
#         )
#     elif cfg.mode == 'infer':
#         test.loc[discourse_ids, ['Ineffective', 'Adequate', 'Effective']] = softmax(
#             pred, axis=-1
#         )
#         submission = pd.read_csv(os.path.join(cfg.comp_data_dir, 'sample_submission.csv'))
#         submission[['Ineffective', 'Adequate', 'Effective']] = test.loc[
#             submission.discourse_id, ['Ineffective', 'Adequate', 'Effective']
#         ].values
#         submission[['discourse_id', 'Ineffective', 'Adequate', 'Effective']].to_csv(
#             'submission.csv', index=False
#         )
#         display(submission.head())
#         submission.to_csv('submission.csv', index=False)


# if __name__ == '__main__':
#     main()
