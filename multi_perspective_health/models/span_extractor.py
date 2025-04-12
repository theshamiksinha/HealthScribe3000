import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
from torchcrf import CRF
from models.base_encoder import BaseEncoder

class SpanExtractorWithCRF(nn.Module):
    def __init__(self, model_name: str, num_tags: int):
        super().__init__()
        self.encoder = BaseEncoder(model_name=model_name)
        self.hidden_size = self.encoder.hidden_size
        self.tag_projection = nn.Linear(self.hidden_size, num_tags)
        self.crf = CRF(num_tags=num_tags, batch_first=True)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        embeddings = self.encoder(input_ids, attention_mask, token_type_ids)
        emissions = self.tag_projection(embeddings)

        if labels is not None:
            loss = -self.crf(emissions, labels, mask=attention_mask.bool(), reduction='mean')
            return loss
        else:
            predictions = self.crf.decode(emissions, mask=attention_mask.bool())
            return predictions
