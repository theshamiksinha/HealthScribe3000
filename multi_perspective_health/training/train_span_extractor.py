# Load the dataset
# Initialize SpanExtractorWithCRF
# Set up optimizer, loss, and metrics
# Run the training loop
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from models.span_extractor import SpanExtractorWithCRF
from data.span_dataset import SpanDataset
import yaml
from utils.metrics import compute_token_f1
import json
from tqdm import tqdm  

def load_config(path='config/config.yaml'):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def train():
    config = load_config()
    model_name = config['model']['encoder_model']
    num_tags = config['model']['num_tags']
    batch_size = config['training']['batch_size']
    epochs = config['training']['num_epochs']
    lr = float(config['training']['learning_rate'])

    with open(config['data']['train_path'], 'r') as f:
        train_data = json.load(f) 
    with open(config['data']['val_path'], 'r') as f:
        val_data = json.load(f) 
    
    # split data for faster training
    # train_data = train_data[:int(len(train_data) * 0.0015)]
    # val_data = val_data[:int(len(val_data) * 0.0015)]

    tokenizer = AutoTokenizer.from_pretrained(config['data']['tokenizer_name'])
    train_dataset = SpanDataset(train_data, tokenizer, config['label_map'])
    val_dataset = SpanDataset(val_data, tokenizer, config['label_map'])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpanExtractorWithCRF(model_name, num_tags).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['labels'].to(device)

            loss = model(input_ids, attention_mask, token_type_ids, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {total_loss / len(train_loader):.4f}")

        # Optionally run validation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} - Validation"):
                input_ids = batch['input_ids'].cuda()
                attention_mask = batch['attention_mask'].cuda()
                labels = batch['labels'].cpu().numpy().tolist()

                preds = model(input_ids, attention_mask)
                all_preds.extend(preds)
                all_labels.extend(labels)

        f1 = compute_token_f1(all_preds, all_labels, config['label_map_reverse'])
        print(f"Validation F1: {f1:.4f}")

if __name__ == "__main__":
    train()