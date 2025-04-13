import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from training.train_classifier import train_classifier, evaluate
from models.perspective_classifier import PerspectiveClassifier
from transformers import AutoTokenizer
from data.data_utils import load_dataset
from data.dataset import PerspectiveClassificationDataset
import torch

def train_or_load_classifier(config):
    train_classifier()  # train and save model
    tokenizer = AutoTokenizer.from_pretrained(config["data"]["tokenizer_name"])
    model = PerspectiveClassifier(
        model_name=config["model"]["classifier"]["encoder_model"],
        num_labels=len(config["perspectives"])
    )
    model.load_state_dict(torch.load(config["training"]["classifer"]["save_dir"]))
    model.eval()
    return model, tokenizer

def predict_perspectives(model, tokenizer, test_data, config):
    dataset = PerspectiveClassificationDataset(
        data=test_data,
        tokenizer_name=config["data"]["tokenizer_name"],
        max_length=config["data"]["max_seq_length"]
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=8)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    all_preds = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids", torch.zeros_like(input_ids)).to(device)

            logits = model(input_ids, attention_mask, token_type_ids)
            preds = (torch.sigmoid(logits) > 0.5).int().cpu().tolist()
            all_preds.extend(preds)

    # Add predicted perspectives to test_data
    perspective_list = list(config["perspectives"].keys())
    for item, pred in zip(test_data, all_preds):
        item["predicted_perspectives"] = [perspective_list[i] for i, val in enumerate(pred) if val == 1]

    return test_data
