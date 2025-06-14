import logging

import pandas as pd
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from argueflow.data.tokenized_dataset import FeedbackPrize2Dataset, collate_fn
from argueflow.utils.dvc_utils import download_data


log = logging.getLogger(__name__)


class FeedbackPrize2DataModule(LightningDataModule):
    """
    PyTorch Lightning DataModule for the Feedback Prize 2 dataset.

    Handles:
    - downloading (via DVC)
    - loading and splitting data
    - preparing train/val/test DataLoaders
    """

    def __init__(self, cfg):
        """
        Args:
            cfg (DictConfig): Hydra configuration object containing paths and training params.
        """
        super().__init__()
        self.cfg = cfg
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        """
        Download data if necessary using DVC.
        Does not split or tokenize (non-mutable operations only).
        """
        log.info("Checking and downloading data if needed...")
        download_data(self.cfg)

    def setup(self, stage=None):
        """
        Split and tokenize data based on training stage.

        Args:
            stage (str): One of ['fit', 'test', None']. Used by Lightning internally.
        """
        if stage == "fit" or stage is None:
            log.info("Loading and preparing training data...")
            df = pd.read_csv(self.cfg.data.processed_data_path)
            full_dataset = FeedbackPrize2Dataset(df, self.cfg)

            val_size = int(self.cfg.train.val_split * len(full_dataset))
            train_size = len(full_dataset) - val_size
            self.train_dataset, self.val_dataset = torch.utils.data.random_split(
                full_dataset,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(self.cfg.train.seed),
            )
        elif stage == "test":
            log.info("Loading and preparing test data...")
            df = pd.read_csv(self.cfg.data.test_data_path)
            self.test_dataset = FeedbackPrize2Dataset(df, self.cfg)

    def train_dataloader(self):
        """Returns DataLoader for training."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.train.batch_size,
            shuffle=True,
            num_workers=self.cfg.train.num_workers,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        """Returns DataLoader for validation."""
        if self.val_dataset is None:
            log.warning("Validation dataset is not set up. Returning empty loader.")
            return None
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.train.batch_size,
            shuffle=False,
            num_workers=self.cfg.train.num_workers,
            collate_fn=collate_fn,
        )
