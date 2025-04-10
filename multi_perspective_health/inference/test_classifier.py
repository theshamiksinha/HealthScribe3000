import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from models.perspective_classifier import PerspectiveClassifier
from data.dataset import PerspectiveClassificationDataset
from utils.metrics import compute_classification_metrics
from config.config import get_config
import json

def test_classifier():
    config = get_config()
    device = torch.device(config["misc"]["device"] if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(config["data"]["tokenizer_name"])

    test_dataset = PerspectiveClassificationDataset(
        data_path=config["data"]["test_path"],
        tokenizer=tokenizer,
        max_length=config["data"]["max_seq_length"]
    )
    test_loader = DataLoader(test_dataset, batch_size=config["training"]["batch_size"], shuffle=False)

    model = PerspectiveClassifier(
        model_name=config["model"]["encoder_model"],
        num_labels=test_dataset.num_labels
    ).to(device)

    classifier_ckpt = "checkpoints/perspective_classifier.pt"  # update if needed
    model.load_state_dict(torch.load(classifier_ckpt, map_location=device))
    model.eval()

    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids, attention_mask, token_type_ids)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    metrics = compute_classification_metrics(all_preds, all_labels)
    print("📊 Perspective Classification Metrics on Test Set:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
        
    # Save pretty predictions to file
    output_path = "outputs/classifier_predictions.jsonl"
    with open(output_path, "w") as f:
        for i, (pred, gold) in enumerate(zip(all_preds, all_labels)):
            entry = {
                "id": test_dataset.examples[i]["id"],
                "text": test_dataset.examples[i]["text"],
                "predicted_label": test_dataset.label_list[pred],
                "gold_label": test_dataset.label_list[gold]
            }
            f.write(json.dumps(entry) + "\n")

    print(f"✅ Saved classification predictions to {output_path}")

if __name__ == "__main__":
    test_classifier()
