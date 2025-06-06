import logging

import pandas as pd
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from argueflow.data.tokenized_dataset import FeedbackPrize2Dataset, collate_fn
from argueflow.utils.dvc_utils import download_data


log = logging.getLogger(__name__)


class FeedbackPrize2DataModule(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        log.info("Checking and downloading data if needed...")

        download_data()

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            log.info("Loading and preparing training data...")
            df = pd.read_csv(self.cfg.data.processed_data_path)
            self.train_dataset = FeedbackPrize2Dataset(df, self.cfg)

            # Stub for validation — currently identical to train
            self.val_dataset = FeedbackPrize2Dataset(df.copy(), self.cfg)

        if stage == "test":
            log.info("Setting up test dataset... (stub)")
            # Stub: could load test.csv later if needed
            self.test_dataset = None

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.train.batch_size,
            shuffle=True,
            num_workers=self.cfg.train.num_workers,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
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

    def test_dataloader(self):
        if self.test_dataset is None:
            log.warning("Test dataset is not set up. Returning empty loader.")
            return None
        return DataLoader(
            self.test_dataset,
            batch_size=self.cfg.train.batch_size,
            shuffle=False,
            num_workers=self.cfg.train.num_workers,
            collate_fn=collate_fn,
        )
