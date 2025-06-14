"""
Inference script for Feedback Prize 2 task.

Loads a trained model from checkpoint and runs prediction on the test set.
"""

import logging
import os
from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from argueflow.data.tokenized_dataset import FeedbackPrize2Dataset, collate_fn
from argueflow.models.fp2_lightning_module import FeedbackPrize2LightningModule


def inference(cfg):
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
    log = logging.getLogger(__name__)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    test_df = pd.read_csv(cfg.data.raw_test_csv)
    log.info(f"Loaded test set with {len(test_df)} rows.")

    cls_token = cfg.model.cls_token
    discourses_df = (
        test_df.groupby("essay_id")["discourse_text"]
        .apply(lambda x: f" {cls_token} ".join(x))
        .reset_index()
        .rename(columns={"discourse_text": "discourses"})
    )

    label_counts = test_df.groupby("essay_id").size().reset_index(name="count")
    label_counts["label_list"] = label_counts["count"].apply(
        lambda n: "|".join(["Adequate"] * n)
    )

    test_df = pd.merge(
        discourses_df, label_counts[["essay_id", "label_list"]], on="essay_id"
    )
    log.info(f"Loaded test set with {len(test_df)} rows.")

    dataset = FeedbackPrize2Dataset(test_df, cfg)

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        collate_fn=collate_fn,
    )

    model = FeedbackPrize2LightningModule.load_from_checkpoint(
        cfg.infer.ckpt_path, cfg=cfg
    )
    model.eval()
    model.to(cfg.trainer.device)

    trainer = pl.Trainer(accelerator=cfg.trainer.accelerator, devices=cfg.trainer.devices)
    predictions = trainer.predict(model, dataloaders=dataloader)

    flat_preds = torch.cat(predictions).tolist()

    label_map_inv = {v: k for k, v in cfg.train.label_map.items()}
    test_df["label_list"] = ["|".join(label_map_inv[i] for i in flat_preds)]

    output_path = Path(cfg.data.output_predictions_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    test_df.to_csv(output_path, index=False)
    log.info(f"Saved predictions to {output_path}")
