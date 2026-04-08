import os
from typing import Tuple, List, Dict

import torch
from transformers import AutoTokenizer

from data.dataset import PerspectiveClassificationDataset
from models.perspective_classifier import PerspectiveClassifier
from training.train_classifier import train_classifier


def train_or_load_classifier(config: Dict) -> Tuple[PerspectiveClassifier, AutoTokenizer]:
    save_dir = config["training"]["classifier"]["save_dir"]

    if not os.path.exists(save_dir):
        print("Model directory not found. Training the model...")
        train_classifier()
    else:
        print("Model directory found. Checking for necessary files...")

        encoder_exists = all([
            os.path.exists(os.path.join(save_dir, "classifier_state_dict.pt")),
            os.path.exists(os.path.join(save_dir, "config.json")),
            os.path.exists(os.path.join(save_dir, "tokenizer_config.json")),
            any(os.path.exists(os.path.join(save_dir, fname)) for fname in ["pytorch_model.bin", "model.safetensors"])
        ])
        classifier_state_dict_exists = os.path.exists(os.path.join(save_dir, "classifier_state_dict.pt"))

        if encoder_exists and classifier_state_dict_exists:
            print("Loading existing model...")
        else:
            print("Missing necessary files. Training the model...")
            train_classifier()

    model = PerspectiveClassifier(model_name="bert-base-uncased", num_labels=5)
    model.encoder.load_transformer(save_dir)

    classifier_state_dict_path = os.path.join(save_dir, "classifier_state_dict.pt")
    if os.path.exists(classifier_state_dict_path):
        print("Loading classifier state dict...")
        model.load_state_dict(torch.load(classifier_state_dict_path, map_location=torch.device('cpu')))
    else:
        print(f"Warning: {classifier_state_dict_path} not found. Skipping state_dict loading.")

    tokenizer = AutoTokenizer.from_pretrained(save_dir)  # type: ignore
    return model, tokenizer


def predict_perspectives(model: PerspectiveClassifier, test_data: List, config: Dict) -> List:
    dataset = PerspectiveClassificationDataset(
        data=test_data,
        tokenizer_name=config["data"]["tokenizer_name"],
        max_length=config["data"]["max_seq_length"]
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=8)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    all_predictions = []
    model.eval()

    with torch.no_grad():
        threshold = config.get("inference", {}).get("threshold", 0.6)

        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids", torch.zeros_like(input_ids)).to(device)

            logits = model(input_ids, attention_mask, token_type_ids)
            probs = torch.sigmoid(logits)

            binary_predictions = (probs > threshold).int().cpu().tolist()
            all_predictions.extend(binary_predictions)

    perspective_list = list(config["perspectives"].keys())
    for item, pred in zip(test_data, all_predictions):
        item["predicted_perspectives"] = [perspective_list[i] for i, val in enumerate(pred) if val == 1]

    return test_data
