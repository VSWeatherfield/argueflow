"""
Inference script for Feedback Prize 2 task.

Loads a trained model from checkpoint and runs prediction on the test set.
"""

import logging
import os

import mlflow
from git import Repo
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from argueflow.data.argue_module import FeedbackPrize2DataModule
from argueflow.models.fp2_lightning_module import FeedbackPrize2LightningModule
from argueflow.utils.dvc_utils import download_data


log = logging.getLogger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def train(cfg: DictConfig):
    """
    Run inference on the raw test set using a trained model checkpoint.

    Args:
        cfg (DictConfig): Hydra configuration containing paths, model params, and loader settings.

    Workflow:
    - Load and preprocess the test CSV
    - Construct discourses and dummy labels
    - Load the dataset and model checkpoint
    - Run inference using a Lightning `Trainer`
    - Save predictions to CSV
    """
    log.info("Checking data availability...")
    download_data(cfg)

    log.info("Initializing Lightning Data Module...")
    dm = FeedbackPrize2DataModule(cfg)

    log.info("Initializing Lightning Module...")
    model = FeedbackPrize2LightningModule(cfg)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_f1",
        mode="max",
        save_top_k=1,
        filename="best_model",
        save_last=True,
    )

    repo = Repo(search_parent_directories=True)
    mlflow.set_tags({"git_commit": repo.head.object.hexsha})

    log.info("Starting training with PyTorch Lightning...")
    trainer = Trainer(
        **cfg.trainer,
        logger=instantiate(cfg.logger),
        callbacks=[checkpoint_callback],
    )

    trainer.fit(model, datamodule=dm)
