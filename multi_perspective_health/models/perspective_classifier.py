# Span classification model
# Input: (question, answer) pair
# Output: The perspective label of the entire answer or answer span
# PerspectiveClassifier needed to predict which perspective(s) apply to the answer
import torch
import torch.nn as nn
from models.base_encoder import BaseEncoder
from transformers import AutoModel

class PerspectiveClassifier(nn.Module):
    def __init__(self, model_name= "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract", num_labels= 5, pos_weight=None):
        """
        Args:
            model_name (str): Name of the pretrained encoder (BioBERT, PubMedBERT, etc.)
            num_labels (int): Number of perspective labels
        """
        super(PerspectiveClassifier, self).__init__()
        self.encoder = BaseEncoder(model_name=model_name)
        self.hidden_size = self.encoder.hidden_size
        self.classifier = nn.Sequential(
                            nn.Linear(self.hidden_size, 256),
                            nn.ReLU(),
                            nn.Dropout(0.3),
                            nn.Linear(256, num_labels)
                        )
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight) if pos_weight is not None else nn.BCEWithLogitsLoss()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        """
        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
            token_type_ids: (batch_size, seq_len)
            labels: (batch_size, num_labels) - multi-hot encoded labels

        Returns:
            If labels provided: (loss, logits)
            Else: logits (batch_size, num_labels)
        """
        # CLS token representation
        pooled_output = self.encoder.get_pooled_output(input_ids, attention_mask, token_type_ids)
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss = self.loss_fn(logits, labels)
            return loss, logits
        else:
            return logits