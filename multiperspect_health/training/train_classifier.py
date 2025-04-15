import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data.data_utils import load_dataset
from transformers import AutoTokenizer
from models.perspective_classifier import PerspectiveClassifier
from data.dataset import PerspectiveClassificationDataset
from utils.metrics import compute_multilabel_metrics
from data.data_utils import load_config, save_predictions_to_json
import json
from tqdm import tqdm  
from collections import Counter
# from modules.perspective_pipeline import predict_perspectives

def train_classifier():
    config = load_config()
    device = torch.device(config["misc"].get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(config["data"]["tokenizer_name"])

    # Load data
    print("Loading datasets...")
    train_data = load_dataset(config['data']['train_path'])
    val_data = load_dataset(config['data']['val_path'])

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
    
    ######################################## data analysis #################################################
    ######################################## data analysis #################################################
    ######################################## data analysis #################################################
    # 1. Count how often each perspective occurs
    label_counter = Counter()
    for ex in train_dataset.examples:  # or full dataset if needed
        label_counter.update(ex["perspectives"])

    # 2. Map label frequencies to perspective order
    perspectives = ["INFORMATION", "SUGGESTION", "CAUSE", "EXPERIENCE", "QUESTION"]
    total = len(train_dataset)
    pos_counts = [label_counter.get(p, 0) for p in perspectives]
    neg_counts = [total - c for c in pos_counts]

    # 3. Compute pos_weight: more weight for rare classes
    pos_weight = torch.tensor([neg / (pos + 1e-5) for pos, neg in zip(pos_counts, neg_counts)], dtype=torch.float)
    
    """######################################DATA ANALYSIS###############################################"""
    ######################################## data analysis #################################################
    ######################################## data analysis #################################################
    ######################################## data analysis #################################################
    label_names = ["INFORMATION", "SUGGESTION", "CAUSE", "EXPERIENCE", "QUESTION"]
    label_counter = torch.zeros(len(label_names))

    for example in train_dataset:
        label_tensor = torch.tensor(example["labels"])  # assuming multi-hot
        label_counter += label_tensor

    print("Label Distribution:")
    for name, count in zip(label_names, label_counter):
        print(f"{name}: {int(count)}")\
    ######################################## data analysis #################################################
    ######################################## data analysis #################################################
    ######################################## data analysis #################################################
        

    train_loader = DataLoader(train_dataset, batch_size=config["training"]["classifier"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["training"]["classifier"]["batch_size"], shuffle=False)

    model = PerspectiveClassifier(
        model_name=config["model"]["classifier"]["encoder_model"],
        num_labels=len(train_dataset.perspectives),
        pos_weight = pos_weight
    ).to(device)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)  # Move the model to the correct device

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["training"]["classifier"]["learning_rate"]))

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)

    best_val_f1 = 0

    for epoch in range(config["training"]["classifier"]["num_epochs"]):
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

        print(f"Epoch {epoch + 1}/{config['training']['classifier']['num_epochs']} - Loss: {total_loss / len(train_loader):.4f}")

        # val_f1 = evaluate(model, val_loader, device, train_dataset.perspectives)
        
    # Save the trained classifier model
    print("✅ Saving the trained PerspectiveClassifier model...\n")
    save_dir = config["training"]["classifier"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    # Save the encoder (Huggingface model part)
    model.encoder.save_transformer(save_dir)
    tokenizer.save_pretrained(save_dir)

    # Save the classifier head and other components
    torch.save(model.state_dict(), os.path.join(save_dir, "classifier_state_dict.pt"))

                
    # test_data = load_dataset(config["data"]["test_path"])
    # predicted_test_data = predict_perspectives(model, tokenizer, test_data, config)  
    # save_predictions_to_json(predicted_test_data)

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
