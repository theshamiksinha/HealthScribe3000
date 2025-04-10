# Input: (question, answer, perspective)
# Output: The exact text spans (token indices or BIO labels) in the answer that express that perspective
# SpanExtractor needed to locate the relevant span(s) within the answer for that predicted perspective

import torch
import torch.nn as nn
from torchcrf import CRF
from models.base_encoder import BaseEncoder

class SpanExtractorWithCRF(nn.Module):
    def __init__(self, model_name: str = "dmis-lab/biobert-base-cased-v1.1", hidden_dim=0, num_tags: int = 3):
        """
        Args:
            model_name (str): Pretrained transformer model name.
            num_tags (int): Number of BIO tags (e.g., B/I/O = 3).
        """
        super(SpanExtractorWithCRF, self).__init__()
        self.encoder = BaseEncoder(model_name=model_name)
        self.hidden_size = self.encoder.hidden_size
        self.num_tags = num_tags

        self.tag_projection = nn.Linear(self.hidden_size, self.num_tags)
        self.crf = CRF(num_tags=self.num_tags, batch_first=True)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        """
        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
            token_type_ids: (batch_size, seq_len)
            labels: (batch_size, seq_len) - gold BIO tag ids

        Returns:
            If labels is provided, returns the negative log-likelihood loss.
            Else, returns the predicted tag sequence (list of list of tag ids).
        """
        embeddings = self.encoder(input_ids, attention_mask, token_type_ids)  # (B, L, H)
        emissions = self.tag_projection(embeddings)  # (B, L, num_tags)

        if labels is not None:
            # Compute loss
            loss = -self.crf(emissions, labels, mask=attention_mask.bool(), reduction='mean')
            return loss
        else:
            # Decode tag sequences
            predictions = self.crf.decode(emissions, mask=attention_mask.bool())
            return predictions
