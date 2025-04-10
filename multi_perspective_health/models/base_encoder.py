import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class BaseEncoder(nn.Module):
    def __init__(self, model_name: str = "dmis-lab/biobert-base-cased-v1.1", output_cls: bool = False):
        super(BaseEncoder, self).__init__()
        self.output_cls = output_cls
        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.config.hidden_size

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        # outputs[0] is the last hidden state: (batch_size, seq_len, hidden_size)
        sequence_output = outputs.last_hidden_state

        if self.output_cls:
            # [CLS] token representation (batch_size, hidden_size)
            cls_output = sequence_output[:, 0, :]
            return cls_output
        return sequence_output
