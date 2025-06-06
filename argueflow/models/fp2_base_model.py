from torch import nn
from transformers import AutoConfig, AutoModel

from argueflow.utils.tokenizer import load_tokenizer


class FeedbackPrize2Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.tokenizer = load_tokenizer(cfg)
        self.cls_token_id = self.tokenizer(cfg.model.cls_token)['input_ids'][1]

        config = AutoConfig.from_pretrained(cfg.model.backbone)
        config.output_hidden_states = False
        config.hidden_dropout_prob = 0.1
        config.attention_probs_dropout_prob = 0.1
        self.config = config

        self.backbone = AutoModel.from_pretrained(cfg.model.backbone, config=config)
        self.backbone.resize_token_embeddings(len(self.tokenizer))

        self.dropouts = nn.ModuleList([nn.Dropout(p) for p in [0.1, 0.2, 0.3, 0.4, 0.5]])
        self.head = nn.Linear(self.config.hidden_size, 3)

        self._init_weights(self.head)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state

        fp2_mask = input_ids == self.cls_token_id

        logits_sum = 0
        for dropout in self.dropouts:
            dropped = dropout(hidden_states)
            logits = self.head(dropped)
            logits_sum += logits

        logits = logits_sum / len(self.dropouts)

        fp2_logits = logits[fp2_mask]

        loss = None
        if labels is not None:
            labels_flat = labels[labels != -100]
            if labels_flat.shape[0] != fp2_logits.shape[0]:
                raise ValueError(
                    f"Mismatch: {labels_flat.shape[0]} labels vs {fp2_logits.shape[0]} logits"
                )
            loss = nn.CrossEntropyLoss()(fp2_logits, labels_flat)

        return loss, fp2_logits
