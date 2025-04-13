import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class BaseEncoder(nn.Module):
    def __init__(self, model_name: str = "bert-base-uncased", output_cls: bool = False):
        super(BaseEncoder, self).__init__()
        self.output_cls = output_cls
        self.model = AutoModel.from_pretrained(model_name)
        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.config.hidden_size
        
    def __call__(self, input_ids, attention_mask=None, token_type_ids=None):
        """Return the full sequence of hidden states (token embeddings)"""
        return self.get_token_embeddings(input_ids, attention_mask, token_type_ids)
        
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
    
    def get_token_embeddings(self, input_ids, attention_mask=None, token_type_ids=None):
        """Get token-level embeddings for sequence tagging"""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        return outputs.last_hidden_state  # Shape: (batch_size, seq_len, hidden_size)
    
    
    def get_pooled_output(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        # Use CLS token or mean pooling
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            return outputs.pooler_output
        else:
            # Mean pooling fallback
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
    def save_transformer(self, save_dir):
        self.model.save_pretrained(save_dir)

    def load_transformer(self, load_dir):
        # Load model weights into existing `self.model`
        self.model = AutoModel.from_pretrained(load_dir)
        self.encoder = self.model  # update self.encoder too