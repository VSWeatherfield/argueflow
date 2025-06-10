import logging
import logging.config

import fire
from hydra import compose, initialize

from argueflow.eval.eval import evaluate
from argueflow.infer.infer import inference
from argueflow.train.train import train
from argueflow.utils.dvc_utils import download_data
from argueflow.utils.logging_utils import setup_logging_from_cfg


log = logging.getLogger(__name__)


class CLI:
    def _run_with_config(self, fn, cfg_path="../../configs", cfg_name="config"):
        """Generic command runner with Hydra Compose + Logging"""
        with initialize(config_path=cfg_path, version_base="1.3"):
            cfg = compose(config_name=cfg_name)
            setup_logging_from_cfg(cfg)
            log.info(f"Running `{fn.__name__}` with config: {cfg_name}")
            return fn(cfg)

    def train(self, cfg_path="../../configs", cfg_name="config"):
        """Train the model"""
        self._run_with_config(train, cfg_path, cfg_name)

    def eval(self, cfg_path="../../configs", cfg_name="config"):
        """Evaluate the model"""
        self._run_with_config(evaluate, cfg_path, cfg_name)

    def infer(self, cfg_path="../../configs", cfg_name="config"):
        """Run inference"""
        self._run_with_config(inference, cfg_path, cfg_name)

    def download_data(self, cfg_path="../../configs", cfg_name="config"):
        """Download data using DVC"""
        self._run_with_config(download_data, cfg_path, cfg_name)

    def prepare_data(self):
        """Data preprocessing placeholder"""
        log.info("No preprocessing function defined yet.")


def main():
    fire.Fire(CLI)
