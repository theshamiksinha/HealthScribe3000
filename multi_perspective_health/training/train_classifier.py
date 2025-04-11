import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from models.perspective_classifier import PerspectiveClassifier
from data.dataset import PerspectiveClassificationDataset
from utils.metrics import compute_multilabel_metrics
import yaml
import json

def load_config(path='config/config.yaml'):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def train_classifier():
    config = load_config()
    device = torch.device(config["misc"].get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(config["data"]["tokenizer_name"])

    # Load JSON data
    with open(config['data']['train_path'], 'r') as f:
        train_data = json.load(f) 
    with open(config['data']['val_path'], 'r') as f:
        val_data = json.load(f) 

    # For faster test runs (adjust/remove for real training)
    train_data = train_data[:int(len(train_data) * 0.0015)]
    val_data = val_data[:int(len(val_data) * 0.0015)]

    train_dataset = PerspectiveClassificationDataset(
        data=train_data,
        tokenizer_name=config["data"]["tokenizer_name"],
        max_length=config["data"]["max_seq_length"]
    )
    val_dataset = PerspectiveClassificationDataset(
        data=val_data,
        tokenizer_name=config["data"]["tokenizer_name"],
        max_length=config["data"]["max_seq_length"]
    )

    train_loader = DataLoader(train_dataset, batch_size=config["training"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["training"]["batch_size"], shuffle=False)

    model = PerspectiveClassifier(
        model_name=config["model"]["pretrained_model"],
        num_labels=len(train_dataset.perspectives)
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)

    best_val_f1 = 0

    for epoch in range(config["training"]["num_epochs"]):
        model.train()
        total_loss = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids", torch.zeros_like(input_ids)).to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            loss, _ = model(input_ids, attention_mask, token_type_ids, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        print(f"Epoch {epoch + 1}/{config['training']['num_epochs']} - Loss: {total_loss / len(train_loader):.4f}")

        val_f1 = evaluate(model, val_loader, device, train_dataset.perspectives)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            if config["training"].get("save_model", False):
                torch.save(model.state_dict(), config["training"]["save_path"])
                print(f"✅ Best model saved (F1 = {val_f1:.4f})")

def evaluate(model, val_loader, device, perspectives):
    model.eval()
    all_preds, all_labels = [], []
    threshold = 0.5  # Threshold for binary classification

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids", torch.zeros_like(input_ids)).to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids, attention_mask, token_type_ids)
            preds = (torch.sigmoid(logits) > threshold).float()

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    metrics = compute_multilabel_metrics(all_preds, all_labels, perspectives)

    print(f"📊 Validation Metrics: Micro F1: {metrics['micro_f1']:.4f}, Macro F1: {metrics['macro_f1']:.4f}")
    for i, perspective in enumerate(perspectives):
        print(f"  - {perspective}: F1 = {metrics['per_class_f1'][i]:.4f}")
    
    return metrics["micro_f1"]

if __name__ == "__main__":
    train_classifier()
