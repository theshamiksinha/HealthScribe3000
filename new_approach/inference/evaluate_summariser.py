import torch
from torch.utils.data import DataLoader
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from bert_score import score as bert_score
import evaluate
from tqdm import tqdm
import json
import os


def evaluate_pegasus_model(model, tokenizer, test_dataset, output_dir="eval_outputs", batch_size=8):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    dataloader = DataLoader(test_dataset, batch_size=batch_size)
    predictions, references = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]

            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=128,
                num_beams=5,
                early_stopping=True,
            )

            decoded_preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

            predictions.extend(decoded_preds)
            references.extend(decoded_labels)

    # === Compute Metrics ===
    results = {}
    
    rouge = evaluate.load("rouge")
    rouge_scores = rouge.compute(predictions=predictions, references=references)
    results.update({f"rouge_{k}": v for k, v in rouge_scores.items()})

    P, R, F1 = bert_score(predictions, references, lang="en")
    results["bertscore_precision"] = P.mean().item()
    results["bertscore_recall"] = R.mean().item()
    results["bertscore_f1"] = F1.mean().item()

    smoothie = SmoothingFunction().method4
    bleu_scores = [sentence_bleu([ref.split()], pred.split(), smoothing_function=smoothie)
                   for pred, ref in zip(predictions, references)]
    results["bleu"] = sum(bleu_scores) / len(bleu_scores)

    meteor_scores = [meteor_score([ref], pred) for pred, ref in zip(predictions, references)]
    results["meteor"] = sum(meteor_scores) / len(meteor_scores)

    # === Save Results ===
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "predictions.txt"), "w") as f:
        for p in predictions:
            f.write(p + "\n")

    with open(os.path.join(output_dir, "references.txt"), "w") as f:
        for r in references:
            f.write(r + "\n")

    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("\n✅ Evaluation Complete. Metrics:")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")

    return results
