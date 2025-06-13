import logging
import os
from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from argueflow.data.tokenized_dataset import FeedbackPrize2Dataset, collate_fn
from argueflow.models.fp2_lightning_module import FeedbackPrize2LightningModule


def inference(cfg):
    log = logging.getLogger(__name__)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    test_df = pd.read_csv(cfg.data.raw_test_csv)
    log.info(f"Loaded test set with {len(test_df)} rows.")

    # tokenizer = load_tokenizer(cfg)
    dataset = FeedbackPrize2Dataset(test_df, cfg)

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        collate_fn=collate_fn,
    )

    model = FeedbackPrize2LightningModule.load_from_checkpoint(
        cfg.infer.checkpoint_path, cfg=cfg
    )
    model.eval()
    model.to(cfg.trainer.accelerator)

    trainer = pl.Trainer(accelerator=cfg.trainer.accelerator, devices=1)
    predictions = trainer.predict(model, dataloaders=dataloader)
    flat_preds = [pred for batch in predictions for pred in batch]

    label_map_inv = {v: k for k, v in cfg.train.label_map.items()}
    test_df["label_list"] = [
        "|".join(label_map_inv[i.item()] for i in row) for row in flat_preds
    ]

    output_path = Path(cfg.data.output_predictions_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    test_df.to_csv(output_path, index=False)
    log.info(f"Saved predictions to {output_path}")
