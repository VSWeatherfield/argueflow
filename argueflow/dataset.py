"""
Dataset module for the Feedback Prize 2 task.

This module handles the conversion of a pandas DataFrame into a PyTorch-compatible
dataset with tokenized input suitable for transformer-based models.

Classes:
    FeedbackPrize2Dataset: PyTorch Dataset that tokenizes text data and encodes labels.

Functions:
    collate_fn: Custom collate function for dynamic padding of label sequences.
"""

import torch
from datasets import Dataset as HFDataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from argueflow.utils import load_tokenizer


class FeedbackPrize2Dataset(Dataset):
    """
    A PyTorch Dataset for tokenizing and preparing the Feedback Prize 2 data.

    Args:
        df (pd.DataFrame): Input dataframe with columns 'discourses' and 'label_list'.
        cfg (Namespace): Configuration object containing model and training parameters.
    """

    def __init__(self, df, cfg):
        self.cfg = cfg

        self.tokenizer = load_tokenizer(cfg)
        self.cls_token_id = self.tokenizer(cfg.model.cls_token)['input_ids'][1]

        self.dataset = HFDataset.from_pandas(df)
        self.dataset = self.dataset.map(
            self.prepare_dataset, batched=False, remove_columns=list(df.columns)
        )

    def prepare_dataset(self, example):
        """
        Tokenizes text and converts label list to integer label IDs.
        """
        tokenized = self.tokenizer(
            example['discourses'],
            padding='max_length',
            truncation=True,
            max_length=self.cfg.model.max_len,
        )
        tokenized['labels'] = [
            self.cfg.training.label_map[x] for x in example['label_list'].split('|')
        ]
        return tokenized

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        return {
            'input_ids': torch.tensor(item['input_ids']),
            'attention_mask': torch.tensor(item['attention_mask']),
            'labels': torch.tensor(item['labels']),
        }


def collate_fn(batch):
    """
    Collates a batch of examples, padding variable-length label sequences.

    Args:
        batch (list): List of dicts from FeedbackPrize2Dataset.

    Returns:
        dict: Batched tensors for input_ids, attention_mask, and padded labels.
    """
    return {
        'input_ids': torch.stack([x['input_ids'] for x in batch]),
        'attention_mask': torch.stack([x['attention_mask'] for x in batch]),
        'labels': pad_sequence(
            [x['labels'] for x in batch], batch_first=True, padding_value=-100
        ),
    }
