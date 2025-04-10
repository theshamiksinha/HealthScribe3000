# Span Dataset for Token-Level BIO Labelling (Multi-Perspective)
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

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

                label_sequence = ['O'] * self.max_len
                for perspective in self.perspective_list:
                    spans = instance.get("labelled_answer_spans", {}).get(perspective, [])
                    for span in spans:
                        start, end = span['label_spans']
                        for i, (start_char, end_char) in enumerate(offset_mapping):
                            if start_char == 0 and end_char == 0:
                                continue  # padding
                            if start_char >= start and end_char <= end:
                                if label_sequence[i] == 'O':
                                    label_sequence[i] = f'B-{perspective}'
                                elif label_sequence[i].startswith('B') or label_sequence[i].startswith('I'):
                                    continue
                                else:
                                    label_sequence[i] = f'I-{perspective}'

                label_ids = [self.label2id.get(tag, 0) for tag in label_sequence]

                token_type_ids = inputs['token_type_ids'][0]

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
        example = self.examples[idx]
        return {
            'input_ids': example['input_ids'],
            'attention_mask': example['attention_mask'],
            'token_type_ids': example['token_type_ids'],
            'labels': example['labels']
        }

