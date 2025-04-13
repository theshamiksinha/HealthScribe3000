import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

# Span Dataset for Token-Level BIO Labelling (Multi-Perspective)

class SpanDataset(Dataset):
    def __init__(self, data, tokenizer, label_map, max_len=512):
        self.data = data
        self.tokenizer = tokenizer
        self.label2id = label_map
        self.id2label = {v: k for k, v in self.label2id.items()}
        self.num_labels = len(self.label2id)
        self.max_len = max_len
        self.perspective_list = ["INFORMATION", "SUGGESTION", "CAUSE", "EXPERIENCE", "QUESTION"]
        self.examples = self.preprocess()

    def preprocess(self):
        examples = []
        for instance in self.data:
            question = instance["question"]
            for answer in instance["answers"]:
                # Process the answer text to extract tokens and spans
                inputs = self.tokenizer(
                    question,
                    answer,
                    truncation=True,
                    padding='max_length',
                    max_length=self.max_len,
                    return_offsets_mapping=True,
                    return_tensors='pt',
                    return_token_type_ids=True
                )
                
                offset_mapping = inputs['offset_mapping'][0].tolist()
                input_ids = inputs['input_ids'][0]
                attention_mask = inputs['attention_mask'][0]
                token_type_ids = inputs['token_type_ids'][0]

                # Initialize all tokens with 'O' tag
                label_sequence = ['O'] * self.max_len
                
                # Find the positions where the answer text starts
                # This helps identify where to look for spans in the tokenized text
                answer_start = 0
                for i, (start_char, end_char) in enumerate(offset_mapping):
                    if token_type_ids[i] == 1 and start_char != 0:  # First token of the answer text
                        answer_start = i
                        break
                
                # Iterate through perspectives and their spans
                for perspective in self.perspective_list:
                    spans = instance.get("labelled_answer_spans", {}).get(perspective, [])
                    
                    for span in spans:
                        # Each span has 'txt' and 'label_spans' (start, end character positions)
                        if 'label_spans' not in span:
                            continue
                            
                        start, end = span['label_spans']
                        span_text = span.get('txt', '')
                        
                        # Find the corresponding tokens for this span
                        in_span = False
                        for i, (start_char, end_char) in enumerate(offset_mapping):
                            if start_char == 0 and end_char == 0:
                                continue  # Skip padding tokens
                                
                            # Check if this token is part of the span
                            if token_type_ids[i] == 1:  # Only consider answer tokens
                                # Adjust character positions relative to the answer text
                                token_in_span = (start_char < end and end_char > start)
                                
                                if token_in_span:
                                    if not in_span:  # First token of the span
                                        label_sequence[i] = f'B-{perspective}'
                                        in_span = True
                                    else:  # Continuation of the span
                                        label_sequence[i] = f'I-{perspective}'
                                else:
                                    in_span = False

                # Convert label sequence to label IDs
                label_ids = [self.label2id.get(tag, 0) for tag in label_sequence]

                examples.append({
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'token_type_ids': token_type_ids,
                    'labels': torch.tensor(label_ids)
                })

        return examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]