import logging
import os

import hydra
import pandas as pd
from omegaconf import DictConfig
from torch.optim import AdamW
from torch.utils.data import DataLoader

from argueflow.dataset import FeedbackPrize2Dataset, collate_fn
from argueflow.model import FeedbackPrize2Model
from argueflow.utils import download_data


log = logging.getLogger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def train_one_epoch(model, dataloader, optimizer, device):
    """
    Train for one epoch, using only `[FP2]` tokens for loss calculation.
    """
    model.train()
    total_loss = 0
    total_fp2 = 0

    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        loss, _ = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_fp2 += labels[labels != -100].size(0)

    avg_loss = total_loss / len(dataloader)
    log.info(f"Train loss: {avg_loss:.4f} — FP2 tokens used: {total_fp2}")
    return avg_loss


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    log.info("Checking data availability...")
    download_data()

    log.info("Loading data...")
    df = pd.read_csv(cfg.data_loading.train_path)

    log.info("Formatting dataset...")
    dataset = FeedbackPrize2Dataset(df, cfg)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        collate_fn=collate_fn,
    )

    log.info("Building model...")
    model = FeedbackPrize2Model(cfg).to(cfg.training.device)

    log.info("Preparing optimizer...")
    optimizer = AdamW(model.parameters(), lr=cfg.training.lr)

    for epoch in range(cfg.training.nepochs):
        log.info(f"\n Epoch {epoch + 1}")
        train_one_epoch(model, dataloader, optimizer, cfg.training.device)

    log.info("Training complete.")


if __name__ == "__main__":
    main()
