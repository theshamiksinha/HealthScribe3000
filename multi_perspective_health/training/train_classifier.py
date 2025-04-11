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
from tqdm import tqdm  # ✅ tqdm added
from collections import Counter
import torch
    
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
    # train_data = train_data[:int(len(train_data) * 0.1)]
    # val_data = val_data[:int(len(val_data) * 0.1)]

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
    # data analysis
    label_names = ["INFORMATION", "SUGGESTION", "CAUSE", "EXPERIENCE", "QUESTION"]
    label_counter = torch.zeros(len(label_names))

    for example in train_dataset:
        label_tensor = torch.tensor(example["labels"])  # assuming multi-hot
        label_counter += label_tensor

    print("Label Distribution:")
    for name, count in zip(label_names, label_counter):
        print(f"{name}: {int(count)}")
        

    train_loader = DataLoader(train_dataset, batch_size=config["training"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["training"]["batch_size"], shuffle=False)

    model = PerspectiveClassifier(
        model_name=config["model"]["pretrained_model"],
        num_labels=len(train_dataset.perspectives)
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["training"]["learning_rate"]))

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)

    best_val_f1 = 0

    for epoch in range(config["training"]["num_epochs"]):
        model.train()
        total_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
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

    tokenizer = val_loader.dataset.tokenizer  # grab tokenizer from dataset
    sample_printed = 0
    max_samples_to_print = 5

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids", torch.zeros_like(input_ids)).to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids, attention_mask, token_type_ids)
            preds = (torch.sigmoid(logits) > threshold).float()

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

            # Print a few samples
            if sample_printed < max_samples_to_print:
                decoded = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
                for i in range(len(decoded)):
                    if sample_printed >= max_samples_to_print:
                        break
                    print("\n📝 Sample", sample_printed + 1)
                    print("Text:", decoded[i])
                    true_labels = [perspectives[j] for j, v in enumerate(labels[i]) if v == 1]
                    pred_labels = [perspectives[j] for j, v in enumerate(preds[i]) if v == 1]
                    print("✅ True Labels:", true_labels)
                    print("🔮 Predicted Labels:", pred_labels)
                    sample_printed += 1

    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    metrics = compute_multilabel_metrics(all_preds, all_labels, perspectives)

    print(f"\n📊 Validation Metrics: Micro F1: {metrics['micro_f1']:.4f}, Macro F1: {metrics['macro_f1']:.4f}")
    for i, perspective in enumerate(perspectives):
        print(f"  - {perspective}: F1 = {metrics['per_class_f1'][i]:.4f}")
    
    return metrics["micro_f1"]


if __name__ == "__main__":
    train_classifier()
