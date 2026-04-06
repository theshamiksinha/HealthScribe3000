import os
import sys

import torch.nn as nn
import torch.nn.functional as F
from models.base_encoder import BaseEncoder

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class PerspectiveClassifier(nn.Module):
    def __init__(self, model_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract", num_labels: int = 5,
                 pos_weight=None):
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

        self.pos_weight = pos_weight  # save this if needed
        self.loss_fn = lambda x, y, pos_weight: F.binary_cross_entropy_with_logits(x, y, pos_weight=pos_weight)

        # self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight) if pos_weight is not None else nn.BCEWithLogitsLoss()

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
        device = input_ids.device  # Get the device of the input tensors (usually, they are on the same device)

        # CLS token representation
        pooled_output = self.encoder.get_pooled_output(input_ids, attention_mask, token_type_ids)
        logits = self.classifier(pooled_output)

        if labels is not None:
            labels = labels.to(device)

            # If pos_weight is used, move it to the same device
            pos_weight = self.pos_weight.to(device)  # Make sure pos_weight is on the same device
            loss = self.loss_fn(logits, labels, pos_weight=pos_weight)  # Assuming pos_weight is used in the loss_fn
            return loss, logits
        else:
            return logits
