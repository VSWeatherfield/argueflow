import logging
import subprocess
from pathlib import Path

from transformers import AutoTokenizer


log = logging.getLogger(__name__)


def download_data():
    """
    Downloads required data from the DVC remote if missing. Checks
    both raw and processed folders, and uses DVC CLI to pull data.
    """
    required_paths = [
        Path("data/raw/train.csv"),
        Path("data/processed/train_prepared.csv"),
    ]

    all_exist = all(path.exists() for path in required_paths)

    if all_exist:
        log.info("All required data already exists.")
        return

    print("Some data is missing — attempting to pull with DVC...")
    try:
        subprocess.run(["dvc", "pull"], check=True)
        log.info("Data successfully downloaded from remote.")
    except FileNotFoundError:
        log.error("DVC is not installed or not in PATH.")
        raise
    except subprocess.CalledProcessError as e:
        log.error("DVC pull failed. Check remote config or credentials.")
        raise e


def load_tokenizer(cfg):
    log.info(f"Loading tokenizer from: {cfg.model.backbone}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.backbone)
    tokenizer.add_special_tokens({'additional_special_tokens': [cfg.model.cls_token]})
    log.info("Tokenizer loaded and [FP2] token added.")
    return tokenizer
