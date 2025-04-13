import torch
from torch.utils.data import DataLoader
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from datasets import load_metric
import evaluate
from bert_score import score as bert_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from tqdm import tqdm
import argparse
import json
import os
from data.data_utils import load_config
from data.llm_dataset import LLMDataset  # Replace with your actual dataset path

def generate_predictions(model, tokenizer, dataloader, device):
    model.eval()
    predictions = []
    references = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating summaries"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=128,
                num_beams=5,
                early_stopping=True
            )

            decoded_preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(batch["labels"], skip_special_tokens=True)

            predictions.extend(decoded_preds)
            references.extend(decoded_labels)

    return predictions, references


def compute_metrics(predictions, references, lang="en"):
    results = {}

    # ROUGE
    rouge = evaluate.load("rouge")
    rouge_scores = rouge.compute(predictions=predictions, references=references)
    results.update({f"rouge_{k}": v for k, v in rouge_scores.items()})

    # BERTScore
    P, R, F1 = bert_score(predictions, references, lang=lang)
    results["bertscore_precision"] = P.mean().item()
    results["bertscore_recall"] = R.mean().item()
    results["bertscore_f1"] = F1.mean().item()

    # BLEU
    smoothie = SmoothingFunction().method4
    bleu_scores = [sentence_bleu([ref.split()], pred.split(), smoothing_function=smoothie)
                   for pred, ref in zip(predictions, references)]
    results["bleu"] = sum(bleu_scores) / len(bleu_scores)

    # METEOR
    meteor_scores = [meteor_score([ref], pred) for pred, ref in zip(predictions, references)]
    results["meteor"] = sum(meteor_scores) / len(meteor_scores)

    return results


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load config and tokenizer
    config = load_config()
    tokenizer = PegasusTokenizer.from_pretrained(args.model_path)
    model = PegasusForConditionalGeneration.from_pretrained(args.model_path).to(device)

    # Load dataset
    test_data = torch.load(args.test_data_path)  # List of dicts
    test_dataset = LLMDataset(test_data, tokenizer, config, mode="test")
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # Generate predictions
    predictions, references = generate_predictions(model, tokenizer, test_loader, device)

    # Compute metrics
    metrics = compute_metrics(predictions, references)

    # Save outputs
    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.output_dir, "predictions.txt"), "w") as f:
        for pred in predictions:
            f.write(pred + "\n")

    with open(os.path.join(args.output_dir, "references.txt"), "w") as f:
        for ref in references:
            f.write(ref + "\n")

    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n🟢 Evaluation complete. Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to fine-tuned Pegasus model")
    parser.add_argument("--test_data_path", type=str, required=True, help="Path to preprocessed test data (torch format)")
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml or JSON")
    parser.add_argument("--output_dir", type=str, default="eval_outputs", help="Directory to save predictions and metrics")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for evaluation")
    args = parser.parse_args()

    main(args)
