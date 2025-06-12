import logging
import os

from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning import Trainer

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

    log.info("Starting training with PyTorch Lightning...")
    trainer = Trainer(
        **cfg.trainer,
        logger=instantiate(cfg.logger),
    )

    trainer.fit(model, datamodule=dm)
