# utils/metrics.py
import numpy as np
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from rouge import Rouge

def compute_token_f1(predictions, gold_labels, id2label=None):
    """Compute token-level F1 score for BIO tagging"""
    # Convert IDs to labels if needed
    if id2label:
        if isinstance(predictions[0][0], int):
            predictions = [[id2label.get(p, 'O') for p in seq] for seq in predictions]
        
        if isinstance(gold_labels[0][0], int):
            gold_labels = [[id2label.get(g, 'O') for g in seq] for seq in gold_labels]
    
    # Flatten predictions and gold labels
    y_pred = [p for seq in predictions for p in seq]
    y_true = [g for seq in gold_labels for g in seq]
    
    # Calculate micro-F1 score
    f1 = f1_score(y_true, y_pred, average='micro')
    
    return f1

def compute_rouge(predictions, references):
    """Compute ROUGE scores for text generation"""
    rouge = Rouge()
    
    # Ensure inputs are non-empty
    valid_pairs = [(p, r) for p, r in zip(predictions, references) 
                   if len(p.strip()) > 0 and len(r.strip()) > 0]
    
    if not valid_pairs:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    
    valid_preds, valid_refs = zip(*valid_pairs)
    
    try:
        scores = rouge.get_scores(valid_preds, valid_refs, avg=True)
        return {
            "rouge1": scores["rouge-1"]["f"],
            "rouge2": scores["rouge-2"]["f"],
            "rougeL": scores["rouge-l"]["f"]
        }
    except Exception as e:
        print(f"Error computing ROUGE: {e}")
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

def compute_multilabel_metrics(predictions, labels, class_names):
    """
    Compute metrics for multi-label classification.
    
    Args:
        predictions: Binary predictions (B, C)
        labels: Ground truth labels (B, C)
        class_names: List of class names
        
    Returns:
        Dictionary of metrics
    """
    # Convert tensors to numpy arrays
    preds_np = predictions.numpy()
    labels_np = labels.numpy()
    
    # Calculate micro and macro metrics
    micro_f1 = f1_score(labels_np, preds_np, average='micro')
    macro_f1 = f1_score(labels_np, preds_np, average='macro')
    micro_precision = precision_score(labels_np, preds_np, average='micro', zero_division=0)
    micro_recall = recall_score(labels_np, preds_np, average='micro', zero_division=0)
    
    # Calculate per-class metrics
    per_class_f1 = f1_score(labels_np, preds_np, average=None, zero_division=0)
    
    return {
        'micro_f1': micro_f1,
        'macro_f1': macro_f1,
        'micro_precision': micro_precision,
        'micro_recall': micro_recall,
        'per_class_f1': per_class_f1
    }
 
def compute_classification_metrics(y_true, y_pred, labels=None):
    """Compute classification metrics for multi-label classification"""
    if isinstance(y_true[0], list) and isinstance(y_pred[0], list):
        # Multi-label classification
        report = classification_report(y_true, y_pred, target_names=labels, zero_division=0)
        
        # Calculate overall F1
        f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        metrics = {
            "f1_micro": f1_micro,
            "f1_macro": f1_macro,
            "report": report
        }
    else:
        # Single-label classification
        report = classification_report(y_true, y_pred, target_names=labels, zero_division=0)
        
        # Calculate overall accuracy and F1
        accuracy = np.mean(np.array(y_true) == np.array(y_pred))
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        metrics = {
            "accuracy": accuracy,
            "f1_macro": f1_macro,
            "report": report
        }
    
    return metrics