import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from models.perspective_classifier import PerspectiveClassifier
from data.dataset import PerspectiveClassificationDataset
from utils.metrics import compute_classification_metrics
from config.config import get_config

def train_classifier():
    config = get_config()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(config["model"]["pretrained_model"])

    train_dataset = PerspectiveClassificationDataset(
        data_path=config["data"]["train_path"],
        tokenizer=tokenizer,
        max_length=config["data"]["max_length"]
    )
    val_dataset = PerspectiveClassificationDataset(
        data_path=config["data"]["val_path"],
        tokenizer=tokenizer,
        max_length=config["data"]["max_length"]
    )

    train_loader = DataLoader(train_dataset, batch_size=config["training"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["training"]["batch_size"], shuffle=False)

    model = PerspectiveClassifier(
        model_name=config["model"]["pretrained_model"],
        num_labels=config["model"]["num_labels"]
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["lr"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)

    best_val_f1 = 0

    for epoch in range(config["training"]["epochs"]):
        model.train()
        total_loss = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            loss = model(input_ids, attention_mask, token_type_ids, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()

        print(f"Epoch {epoch + 1}/{config['training']['epochs']} - Loss: {total_loss / len(train_loader):.4f}")

        val_f1 = evaluate(model, val_loader, device)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), config["training"]["save_path"])
            print(f"✅ Best model saved (F1 = {val_f1:.4f})")

def evaluate(model, val_loader, device):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids, attention_mask, token_type_ids)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    metrics = compute_classification_metrics(all_preds, all_labels)
    print(f"📊 Validation Metrics: Acc: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")
    return metrics["f1"]

if __name__ == "__main__":
    train_classifier()
