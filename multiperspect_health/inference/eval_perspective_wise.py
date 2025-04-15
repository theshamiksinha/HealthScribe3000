from collections import defaultdict
import torch
from torch.utils.data import DataLoader
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
from bert_score import score as bertscore
from tabulate import tabulate
import tqdm

def evaluate_perspective_wise(model, tokenizer, dataset, all_perspectives=None):
    print("Generating perspective-wise predictions...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)

    # Wrap dataset in a DataLoader
    dataloader = DataLoader(dataset, batch_size=1)

    # Store predictions and refs per perspective
    results = defaultdict(lambda: {"preds": [], "refs": []})

    for batch in tqdm.tqdm(dataloader):
        # Unwrap batch (since batch_size=1)
        batch = {k: v.squeeze(0).to(device) if isinstance(v, torch.Tensor) else v[0] for k, v in batch.items()}
        
        input_ids = batch["input_ids"].unsqueeze(0)
        attention_mask = batch["attention_mask"].unsqueeze(0)
        if "labels" not in batch or batch["labels"] is None:
            continue  # skip this batch if labels are missing
        label_ids = batch["labels"].unsqueeze(0)

        perspective = batch["perspective"]

        with torch.no_grad():
            generated_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=512)

        decoded_pred = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        decoded_label = tokenizer.decode(label_ids[0][label_ids[0] != -100], skip_special_tokens=True)

        results[perspective]["preds"].append(decoded_pred.strip())
        results[perspective]["refs"].append(decoded_label.strip())

    all_perspectives = all_perspectives or sorted(results.keys())
    table = []

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    smoothie = SmoothingFunction().method4

    for perspective in all_perspectives:
        preds = results[perspective]["preds"]
        refs = results[perspective]["refs"]

        if not preds:
            table.append([perspective] + ["N/A"] * 9)
            continue

        rouge1_r, rouge1_f = [], []
        rouge2_r, rouge2_f = [], []
        rougeL_r, rougeL_f = [], []
        bleu_scores = []
        meteor_scores = []

        for pred, ref in zip(preds, refs):
            scores = scorer.score(ref, pred)
            rouge1_r.append(scores["rouge1"].recall)
            rouge1_f.append(scores["rouge1"].fmeasure)
            rouge2_r.append(scores["rouge2"].recall)
            rouge2_f.append(scores["rouge2"].fmeasure)
            rougeL_r.append(scores["rougeL"].recall)
            rougeL_f.append(scores["rougeL"].fmeasure)

            bleu_scores.append(sentence_bleu([word_tokenize(ref)], word_tokenize(pred), smoothing_function=smoothie))
            meteor_scores.append(meteor_score([word_tokenize(ref)], word_tokenize(pred)))

        P, R, F1 = bertscore(preds, refs, lang="en", verbose=False)
        bert_f1 = F1.mean().item()

        def truncate(x, decimals=2):
            factor = 10 ** decimals
            return int(x * factor) / factor

        table.append([
            perspective,
            truncate(sum(rouge1_r)/len(rouge1_r)), truncate(sum(rouge1_f)/len(rouge1_f)),
            truncate(sum(rouge2_r)/len(rouge2_r)), truncate(sum(rouge2_f)/len(rouge2_f)),
            truncate(sum(rougeL_r)/len(rougeL_r)), truncate(sum(rougeL_f)/len(rougeL_f)),
            truncate(sum(bleu_scores)/len(bleu_scores)),
            truncate(sum(meteor_scores)/len(meteor_scores)),
            truncate(bert_f1)
        ])


    print("\n" + tabulate(
        table,
        headers=[
            "Perspective", "ROUGE-1 Recall", "ROUGE-1 F1",
            "ROUGE-2 Recall", "ROUGE-2 F1",
            "ROUGE-L Recall", "ROUGE-L F1",
            "BLEU", "METEOR", "BERTScore F1"
        ],
        floatfmt=".6f",
        tablefmt="fancy_grid"
    ))
