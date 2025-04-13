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
    save_dir = config["training"]["classifier"]["save_dir"]
 
    if not os.path.exists(save_dir):
        print("Model directory not found. Training the model...")
        train_classifier()  # Train and save the model
    else:
        print("Model directory found. Checking for necessary files...")
 
        encoder_exists = all([
            os.path.exists(os.path.join(save_dir, "classifier_state_dict.pt")),
            os.path.exists(os.path.join(save_dir, "config.json")),
            os.path.exists(os.path.join(save_dir, "tokenizer_config.json")),  # optional but nice
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
    
    # Check if the classifier state dict exists and load it
    classifier_state_dict_path = os.path.join(save_dir, "classifier_state_dict.pt")
    if os.path.exists(classifier_state_dict_path):
        print("Loading classifier state dict...")
        model.load_state_dict(torch.load(classifier_state_dict_path))
    else:
        print(f"Warning: {classifier_state_dict_path} not found. Skipping state_dict loading.")
    
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(save_dir)

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
