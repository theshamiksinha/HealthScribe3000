
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from torch.utils.data import Dataset
from transformers import AutoTokenizer
import torch

class PerspectiveClassificationDataset(Dataset):
    def __init__(self, data, tokenizer_name="bert-base-uncased", max_length=512):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.perspectives = ["INFORMATION", "SUGGESTION", "CAUSE", "EXPERIENCE", "QUESTION"]
        self.perspective_to_idx = {p: i for i, p in enumerate(self.perspectives)}
        
        # Flatten dataset: create one item per (answer) item
        self.examples = []
        for item in data:
            question = item["question"]
            answers = item["answers"]
            labelled_spans = item.get("labelled_answer_spans", {})
            
            # For each answer, check which perspectives it contains
            for answer in answers:
                answer_start = item["raw_text"].find(answer)
                answer_end = answer_start + len(answer)
                
                present_perspectives = set()
                for p, spans in labelled_spans.items():
                    for span in spans:
                        span_start, span_end = span["label_spans"]
                        if answer_start <= span_start < answer_end or answer_start < span_end <= answer_end:
                            present_perspectives.add(p)
                
                self.examples.append({
                    # "question": question,
                    "answer": answer,
                    "perspectives": list(present_perspectives)
                })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        text = example["answer"]
        # text = example["question"] + " " + example["answer"]
        
        encoded = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)
        token_type_ids = encoded.get("token_type_ids", torch.zeros_like(input_ids)).squeeze(0)

        label = torch.zeros(len(self.perspectives))
        for perspective in example["perspectives"]:
            if perspective in self.perspective_to_idx:
                label[self.perspective_to_idx[perspective]] = 1.0

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "labels": label
        }
