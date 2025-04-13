import torch
from collections import defaultdict
from tqdm import tqdm
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
from bert_score import score as bertscore
import pandas as pd

def evaluate_perspective_wise(model, tokenizer, dataset):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    perspective_predictions = defaultdict(list)
    perspective_references = defaultdict(list)

    print("\nGenerating perspective-wise predictions...")
    for batch in tqdm(dataset):
        input_ids = batch["input_ids"].unsqueeze(0).to(device)
        attention_mask = batch["attention_mask"].unsqueeze(0).to(device)
        label_ids = batch["labels"].unsqueeze(0).to(device)
        perspective = batch["perspective"]

        with torch.no_grad():
            generated_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=512)

        pred = tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()
        label = tokenizer.decode(label_ids[0][label_ids[0] != -100], skip_special_tokens=True).strip()

        perspective_predictions[perspective].append(pred)
        perspective_references[perspective].append(label)

    def compute_metrics(predictions, references):
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        r1_f, r1_r, r2_f, r2_r, rl_f, rl_r = [], [], [], [], [], []

        for pred, ref in zip(predictions, references):
            scores = scorer.score(ref, pred)
            r1_f.append(scores["rouge1"].fmeasure)
            r1_r.append(scores["rouge1"].recall)
            r2_f.append(scores["rouge2"].fmeasure)
            r2_r.append(scores["rouge2"].recall)
            rl_f.append(scores["rougeL"].fmeasure)
            rl_r.append(scores["rougeL"].recall)

        smoothie = SmoothingFunction().method4
        bleu = [sentence_bleu([word_tokenize(ref)], word_tokenize(pred), smoothing_function=smoothie)
                for pred, ref in zip(predictions, references)]

        meteor = [meteor_score([word_tokenize(ref)], word_tokenize(pred))
                  for pred, ref in zip(predictions, references)]

        P, R, F1 = bertscore(predictions, references, lang="en", verbose=False)

        return {
            "ROUGE-1 Recall": sum(r1_r) / len(r1_r),
            "ROUGE-1 F1":     sum(r1_f) / len(r1_f),
            "ROUGE-2 Recall": sum(r2_r) / len(r2_r),
            "ROUGE-2 F1":     sum(r2_f) / len(r2_f),
            "ROUGE-L Recall": sum(rl_r) / len(rl_r),
            "ROUGE-L F1":     sum(rl_f) / len(rl_f),
            "BLEU":           sum(bleu) / len(bleu),
            "METEOR":         sum(meteor) / len(meteor),
            "BERTScore F1":   F1.mean().item()
        }

    # Compute and display perspective-wise scores
    print("\n==== Perspective-wise Evaluation Results ====")
    results = []

    for perspective in perspective_predictions:
        preds = perspective_predictions[perspective]
        refs = perspective_references[perspective]
        scores = compute_metrics(preds, refs)
        row = {"Perspective": perspective}
        row.update(scores)
        results.append(row)

    df = pd.DataFrame(results)
    print(df.to_markdown(index=False))

    return df
