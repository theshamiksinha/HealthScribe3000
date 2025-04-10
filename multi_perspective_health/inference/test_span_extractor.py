import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from ..models.span_extractor import SpanExtractorWithCRF
from data.span_dataset import SpanExtractionDataset
from utils.metrics import compute_span_metrics
from config.config import get_config
import json

def test_span_extractor():
    config = get_config()
    device = torch.device(config["misc"]["device"] if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(config["data"]["tokenizer_name"])

    test_dataset = SpanExtractionDataset(
        data_path=config["data"]["test_path"],
        tokenizer=tokenizer,
        max_length=config["data"]["max_seq_length"]
    )
    test_loader = DataLoader(test_dataset, batch_size=config["training"]["batch_size"], shuffle=False)

    model = SpanExtractorWithCRF(
        model_name=config["model"]["encoder_model"],
        hidden_dim=config["model"]["hidden_dim"],
        dropout=config["model"]["dropout"],
        use_crf=config["model"]["use_crf"],
        num_labels=test_dataset.num_labels
    ).to(device)

    model.load_state_dict(torch.load(config["training"]["save_path"], map_location=device))
    model.eval()

    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["labels"].to(device)

            preds = model.predict(input_ids, attention_mask, token_type_ids)

            all_preds.extend(preds)
            all_labels.extend(labels.cpu().tolist())

    metrics = compute_span_metrics(all_preds, all_labels, test_dataset.label_list)
    print("📊 Span Extraction Metrics on Test Set:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
        

    # Save pretty predictions to file
    output_path = "outputs/span_predictions.jsonl"
    with open(output_path, "w") as f:
        for i, (pred, gold) in enumerate(zip(all_preds, all_labels)):
            tokens = test_dataset.examples[i]["tokens"]
            pred_tags = [test_dataset.label_list[tag] for tag in pred]
            gold_tags = [test_dataset.label_list[tag] for tag in gold]

            entry = {
                "id": test_dataset.examples[i]["id"],
                "tokens": tokens,
                "predicted_tags": pred_tags,
                "gold_tags": gold_tags
            }
            f.write(json.dumps(entry) + "\n")

    print(f"✅ Saved span predictions to {output_path}")


if __name__ == "__main__":
    test_span_extractor()
