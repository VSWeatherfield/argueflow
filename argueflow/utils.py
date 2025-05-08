from transformers import AutoTokenizer


def load_tokenizer(cfg):
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.backbone)
    tokenizer.add_special_tokens({'additional_special_tokens': [cfg.model.cls_token]})
    return tokenizer
