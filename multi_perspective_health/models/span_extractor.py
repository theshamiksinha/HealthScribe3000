import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Input: (question, answer, perspective)
# Output: The exact text spans (token indices or BIO labels) in the answer that express that perspective
import torch
import torch.nn as nn
from torchcrf import CRF
from models.base_encoder import BaseEncoder

class SpanExtractorWithCRF(nn.Module):
    def __init__(self, model_name: str = "dmis-lab/biobert-base-cased-v1.1", num_tags: int = 11):
        """
        Args:
            model_name (str): Pretrained transformer model name.
            num_tags (int): Number of BIO tags (B-PERSPECTIVE, I-PERSPECTIVE, O)
                            For 5 perspectives + O tag = 11 tags
        """
        super(SpanExtractorWithCRF, self).__init__()
        self.encoder = BaseEncoder(model_name=model_name)
        self.hidden_size = self.encoder.hidden_size
        self.num_tags = num_tags

        self.dropout = nn.Dropout(0.1)
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
        # Get token-level representations from the encoder
        token_embeddings = self.encoder.get_token_embeddings(
            input_ids, attention_mask, token_type_ids
        )  # (B, L, H)
        
        token_embeddings = self.dropout(token_embeddings)
        emissions = self.tag_projection(token_embeddings)  # (B, L, num_tags)

        if labels is not None:
            # Compute CRF loss (negative log-likelihood)
            # Make sure mask is boolean
            mask = attention_mask.bool()
            loss = -self.crf(emissions, labels, mask=mask, reduction='mean')
            return loss
        else:
            # Decode tag sequences
            mask = attention_mask.bool()
            predictions = self.crf.decode(emissions, mask=mask)
            return predictions
            
    # Helper method to get tag names from predicted tag IDs
    def get_tag_names(self, tag_ids, id2label):
        return [[id2label[tag_id] for tag_id in seq] for seq in tag_ids]
    
    def predict(self, input_ids, attention_mask):
        self.eval()  # Set the model in evaluation mode
        with torch.no_grad():
            outputs = self(input_ids, attention_mask=attention_mask)
            logits = outputs[1] if isinstance(outputs, tuple) else outputs  # Check if the output is a tuple
            logits = logits[0] if isinstance(logits, tuple) else logits  # Ensure logits is a tensor

            # Apply argmax to get predicted tag indices
            predicted_tag_idxs = torch.argmax(logits, dim=-1).cpu().numpy()
        
        return predicted_tag_idxs