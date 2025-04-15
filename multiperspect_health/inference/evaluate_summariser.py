import os
import torch
from tqdm import tqdm
from transformers import pipeline
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
from bert_score import score as bertscore
from tabulate import tabulate
import nltk
nltk.download("punkt")


def evaluate_pegasus_model(model, tokenizer, dataset, output_dir="eval_after_training"):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    predictions, references = [], []

    print("Generating predictions...")
    for batch in tqdm(dataset):
        input_ids = batch["input_ids"].unsqueeze(0).to(device)
        attention_mask = batch["attention_mask"].unsqueeze(0).to(device)
        if "labels" not in batch or batch["labels"] is None:
            continue  # skip this batch if labels are missing

        label_ids = batch["labels"].unsqueeze(0).to(device)


        with torch.no_grad():
            generated_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=512)

        decoded_pred = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        decoded_label = tokenizer.decode(label_ids[0][label_ids[0] != -100], skip_special_tokens=True)

        predictions.append(decoded_pred.strip())
        references.append(decoded_label.strip())

    # ROUGE
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rouge_scores = {"rouge1": [], "rouge2": [], "rougeL": []}

    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        for key in rouge_scores:
            rouge_scores[key].append((scores[key].recall, scores[key].fmeasure))

    # Compute average recall and f1 for each rouge type
    table_data = []
    for key in ["rouge1", "rouge2", "rougeL"]:
        recalls = [r for r, f in rouge_scores[key]]
        f1s = [f for r, f in rouge_scores[key]]
        avg_recall = sum(recalls) / len(recalls)
        avg_f1 = sum(f1s) / len(f1s)
        table_data.append([key.upper(), f"{avg_recall:.4f}", f"{avg_f1:.4f}"])

    # BLEU
    smoothie = SmoothingFunction().method4
    bleu_scores = [
        sentence_bleu([word_tokenize(ref)], word_tokenize(pred), smoothing_function=smoothie)
        for pred, ref in zip(predictions, references)
    ]
    avg_bleu = sum(bleu_scores) / len(bleu_scores)

    # METEOR
    meteor_scores = [
        meteor_score([word_tokenize(ref)], word_tokenize(pred))
        for pred, ref in zip(predictions, references)
    ]
    avg_meteor = sum(meteor_scores) / len(meteor_scores)

    # BERTScore
    print("Calculating BERTScore...")
    P, R, F1 = bertscore(predictions, references, lang="en", verbose=True)
    avg_bertscore_f1 = F1.mean().item()

    # Print results
    print("\n==== Evaluation Results ====")
    print(tabulate(table_data, headers=["ROUGE Type", "Recall", "F1"], tablefmt="fancy_grid"))
    print(f"\nBLEU:       {avg_bleu:.4f}")
    print(f"METEOR:     {avg_meteor:.4f}")
    print(f"BERTScore F1: {avg_bertscore_f1:.4f}")

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "metrics.txt"), "w") as f:
        f.write(tabulate(table_data, headers=["ROUGE Type", "Recall", "F1"], tablefmt="grid"))
        f.write(f"\n\nBLEU:       {avg_bleu:.4f}\n")
        f.write(f"METEOR:     {avg_meteor:.4f}\n")
        f.write(f"BERTScore F1: {avg_bertscore_f1:.4f}\n")
    