# Dataset class and dataloaders
# General Dataset (for perspective classification)
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class PerspectiveClassificationDataset(Dataset):
    def __init__(self, data, tokenizer_name="bert-base-uncased", max_length=512):
        self.data = data
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        question = item["question"]
        answers = item["answers"]  # list of answers
        perspective_labels = item["labelled_answer_spans"]

        inputs, labels = [], []
        for perspective, spans in perspective_labels.items():
            for span in spans:
                text = question + " " + span["txt"]
                encoded = self.tokenizer(
                    text,
                    truncation=True,
                    padding="max_length",
                    max_length=self.max_length,
                    return_tensors="pt"
                )
                inputs.append(encoded)
                labels.append(perspective)

        return inputs, labels

 
