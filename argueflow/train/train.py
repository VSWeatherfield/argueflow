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
