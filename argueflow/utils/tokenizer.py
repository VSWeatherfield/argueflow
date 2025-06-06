import logging

from transformers import AutoTokenizer


log = logging.getLogger(__name__)


def load_tokenizer(cfg):
    log.info(f"Loading tokenizer from: {cfg.model.backbone}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.backbone)
    tokenizer.add_special_tokens({'additional_special_tokens': [cfg.model.cls_token]})
    log.info("Tokenizer loaded and [FP2] token added.")
    return tokenizer
