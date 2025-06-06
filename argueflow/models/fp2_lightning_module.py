import logging

import pytorch_lightning as pl
from torch.optim import AdamW

from argueflow.models.fp2_base_model import FeedbackPrize2Model


log = logging.getLogger(__name__)


class FeedbackPrize2LightningModule(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = FeedbackPrize2Model(cfg)
        self.save_hyperparameters(cfg)

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        loss, _ = self(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        # Optional: log loss per step
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        fp2_count = (labels != -100).sum().item()
        self.log("fp2_tokens", fp2_count, on_step=True, prog_bar=False)

        return loss

    def configure_optimizers(self):
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.cfg.train.lr,
        )
        return optimizer
