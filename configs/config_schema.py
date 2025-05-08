from dataclasses import dataclass
from typing import Dict


@dataclass
class ModelConfig:
    backbone: str
    max_len: int
    cls_token: str


@dataclass
class DataLoadingConfig:
    raw_train_csv: str
    raw_essay_folder: str
    processed_data_path: str
    test_csv: str
    sample_submission: str


@dataclass
class TrainingConfig:
    batch_size: int
    num_workers: int
    nepochs: int
    lr: float
    device: str
    label_map: Dict[str, int]


@dataclass
class LoggingConfig:
    wandb: bool
    log_dir: str
    verbosity: str


@dataclass
class Config:
    model: ModelConfig
    data_loading: DataLoadingConfig
    training: TrainingConfig
    logging: LoggingConfig
