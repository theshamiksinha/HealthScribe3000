
import json
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader


"""Load data from the given filepath, using load json"""
def load_data_from_json(filepath):
    
    with open(filepath, "r", encoding="utf-8") as file:
        return json.load(file)
    

        
train_data = load_data_from_json("Baseline_Folder/puma_dataset/train.json")
test_data = load_data_from_json("Baseline_Folder/puma_dataset/test.json")
val_data = load_data_from_json("Baseline_Folder/puma_dataset/valid.json")

# Print first entry from train_data to inspect structure
# print(json.dumps(test_data[0], indent=4))  # Pretty-print first data entry

PERSPECTIVE_DEFINITION = {
    "INFORMATION_SUMMARY" : "Defined as knowledge about diseases, disorders, and health-related facts, providing insights into symptoms and diagnosis.",
    "CAUSE_SUMMARY" : "Defined as reasons responsible for the occurrence of a particular medical condition, symptom, or disease",
    "SUGGESTION_SUMMARY" : "Defined as advice or recommendations to assist users in making informed medical decisions, solving problems, or improving health issues.",
    "QUESTION_SUMMARY" : "Defined as inquiry made for deeper understanding.",
    "EXPERIENCE_SUMMARY" : "Defined as individual experiences, anecdotes, or firsthand insights related to health, medical treatments, medication usage, and coping strategies."
}

PERSPECTIVE_TONE = {
    "INFORMATION_SUMMARY" : "Informative, Educational.",
    "CAUSE_SUMMARY" : "Explanatory, Causal.",
    "SUGGESTION_SUMMARY" : "Advisory, Recommending.",
    "QUESTION_SUMMARY" : "Seeking Understanding.",
    "EXPERIENCE_SUMMARY" : "Personal, Narrative."
}

PERSPECTIVE_START_PHRASE = {
    "INFORMATION_SUMMARY" : "For information purposes...",
    "CAUSE_SUMMARY" : "Some of the causes...",
    "SUGGESTION_SUMMARY" : "It is suggested...",
    "QUESTION_SUMMARY" : "It is inquired...",
    "EXPERIENCE_SUMMARY" : "In user’s experience..."
}


"""
Custom dataset to dynamically create prompts and tokenize data on the fly.
"""
class PerspectiveDataset(Dataset):
    
    
    def __init__(self, data, tokenizer, max_input_length=512, max_output_length=200):
        
        self.data = data
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        
        self.flattened_data = []
        for entry in self.data:
            question = entry['question']
            answers = " ".join(entry['answers']) if entry['answers'] else ""
            
            for perspective, summary in entry['labelled_summaries'].items():
                self.flattened_data.append((entry, perspective, summary))
    
    
    def __len__(self):
        
        return len(self.flattened_data)
    
    
    def __getitem__(self, idx): 
        
        entry, perspective, summary = self.flattened_data[idx] 
        question = entry["question"]
        answers = " ".join(entry["answers"]) if entry["answers"] else ""
        
        perspective_def = PERSPECTIVE_DEFINITION.get(perspective, "")
        tone = PERSPECTIVE_TONE.get(perspective, "Neutral.")
        starting_phrase = PERSPECTIVE_START_PHRASE.get(perspective, "")
        
        prompt = (
            f"Summarize the following content according to Perspective: {perspective};\n"
            f"{perspective} Definition: {perspective_def};\n"
            f"Begin Summary with: {starting_phrase};\n"
            f"Tone of summary: {tone};\n"
            f"Content to summarize: {answers}\n"
            f"Associated question: {question}\n"
        )
        
        # tokenizing the input prompt and the target summary
        input = self.tokenizer(
            prompt,
            max_length = self.max_input_length,
            truncation = True,
            padding = "max_length",
        )
        
        labels = self.tokenizer(
            summary,
            max_length=self.max_output_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )


        input["labels"] = labels["input_ids"].squeeze(0)  # Remove extra dimension
        input = {key: torch.tensor(val).squeeze(0) for key, val in input.items()}
        return input  # Convert batch format to tensor

# Load tokenizer
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Create dataset instance
train_dataset = PerspectiveDataset(train_data, tokenizer)
valid_dataset = PerspectiveDataset(test_data, tokenizer)

# Check dataset length
print(f"Original dataset size: {len(train_data)}")
print(f"Expanded dataset size (with all perspectives): {len(train_dataset)}")

# Print the first few samples
for i in range(5):
    sample = train_dataset[i]
    print(f"\nSample {i+1}:")
    print(f"Input: {tokenizer.decode(sample['input_ids'], skip_special_tokens=True)}")
    print(f"Summary: {tokenizer.decode(sample['labels'], skip_special_tokens=True)}")

# Create DataLoaders for batching
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=8, shuffle=False)

# Get the first batch
batch = next(iter(train_dataloader))

# Print batch details
print("\nBatch Shape:")
print(f"Input IDs Shape: {batch['input_ids'].shape}")  # Should be [batch_size, max_input_length]
print(f"Labels Shape: {batch['labels'].shape}")        # Should be [batch_size, max_target_length]