from torch import nn
from transformers import AutoModel


class FeedbackPrize2Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(cfg.backbone, config=cfg.config)
        self.backbone.resize_token_embeddings(len(cfg.tokenizer))
        self.dropout = nn.Dropout(0.3)
        self.head = nn.Linear(cfg.config.hidden_size, 3)

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        logits = self.head(self.dropout(pooled_output))
        return logits
