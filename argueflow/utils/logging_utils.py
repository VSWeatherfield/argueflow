import logging

from omegaconf import OmegaConf


def setup_logging_from_cfg(cfg):
    """
    Initialize logging using Hydra-style config in `cfg.python_logging`
    """
    if "python_logging" in cfg:
        logging_config = OmegaConf.to_container(cfg.python_logging, resolve=True)
        logging.config.dictConfig(logging_config)
    else:
        logging.basicConfig(level=logging.INFO)
        logging.getLogger(__name__).warning(
            "No python_logging config found — using basicConfig."
        )
