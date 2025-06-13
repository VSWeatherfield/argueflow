import logging

import pytorch_lightning as pl
import torch
import torchmetrics
from torch.optim import AdamW

from argueflow.models.fp2_base_model import FeedbackPrize2Model


log = logging.getLogger(__name__)


class FeedbackPrize2LightningModule(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = FeedbackPrize2Model(cfg)
        self.save_hyperparameters(cfg)

        self.val_f1 = torchmetrics.F1Score(
            task="multiclass",
            num_classes=cfg.train.num_classes,
            average="macro",
            ignore_index=-100,
        )

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        loss, _ = self(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True
        )

        # fp2_count = (labels != -100).sum().item()
        # self.log("fp2_tokens", fp2_count, on_step=True, prog_bar=False)

        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        loss, logits = self(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        self.log(
            "val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True
        )

        preds = torch.argmax(logits, dim=-1)
        labels_flat = labels[labels != -100]

        self.val_f1.update(preds, labels_flat)

    def on_validation_epoch_end(self):
        f1 = self.val_f1.compute()
        self.log("val_f1", f1, prog_bar=True, sync_dist=True)
        self.val_f1.reset()

    def configure_optimizers(self):
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.cfg.train.lr,
        )
        return optimizer
