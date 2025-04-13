# utils/metrics.py

from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import classification_report
# Modified token F1 function for span extraction
def compute_token_f1(predictions, gold_labels, id2label):
    """
    Compute token-level F1 score for BIO tagging.
    
    Args:
        predictions: List of predicted tag sequences
        gold_labels: List of gold tag sequences
        id2label: Mapping from tag IDs to tag names
        
    Returns:
        F1 score and a detailed classification report
    """
    # Flatten predictions and labels, but only include non-padding tokens
    y_true = []
    y_pred = []
    
    for pred_seq, true_seq in zip(predictions, gold_labels):
        # Convert IDs to tag names if needed
        if isinstance(pred_seq[0], int):
            pred_seq = [id2label[p] for p in pred_seq]
        if isinstance(true_seq[0], int):
            true_seq = [id2label[t] for t in true_seq]
        
        y_true.extend(true_seq)
        y_pred.extend(pred_seq)
    
    # Generate classification report
    report = classification_report(y_true, y_pred)
    
    # Convert to string labels if needed and filter out padding tokens
    # Calculate F1 scores for each tag class and average
    
    # Calculate micro F1 score (across all tokens and classes)
    f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
    
    return f1_micro, report


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
    
    
def compute_span_metrics(pred_spans, true_spans):
    """
    Compute precision, recall, and F1 score for span extraction task.

    Arguments:
        pred_spans: List of predicted spans (e.g., [(start1, end1), (start2, end2), ...])
        true_spans: List of true spans (e.g., [(start1, end1), (start2, end2), ...])

    Returns:
        precision: Precision score
        recall: Recall score
        f1: F1 score
    """
    # Flatten the spans into a sequence of 0's and 1's for precision/recall/F1 computation
    pred_labels = [1 if span in pred_spans else 0 for span in true_spans]
    true_labels = [1 if span in true_spans else 0 for span in true_spans]

    # Compute precision, recall, and F1 score
    precision = precision_score(true_labels, pred_labels)
    recall = recall_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels)

    return precision, recall, f1

def compute_seqeval_metrics(true_labels, pred_labels):
    """
    Args:
        true_labels: List of lists of true BIO tags (e.g. [['B-INFO', 'I-INFO', 'O'], ...])
        pred_labels: List of lists of predicted BIO tags

    Returns:
        A dict of precision, recall, f1
    """
    return {
        "precision": precision_score(true_labels, pred_labels),
        "recall": recall_score(true_labels, pred_labels),
        "f1": f1_score(true_labels, pred_labels)
    }

def detailed_report(true_labels, pred_labels):
    """
    Returns a classification report with per-class metrics
    """
    return classification_report(true_labels, pred_labels, digits=4)

def compute_token_accuracy(true_labels, pred_labels):
    """
    Computes plain token-level accuracy.
    """
    total, correct = 0, 0
    for true_seq, pred_seq in zip(true_labels, pred_labels):
        for t, p in zip(true_seq, pred_seq):
            total += 1
            if t == p:
                correct += 1
    return correct / total if total > 0 else 0.0


def compute_classification_metrics(preds, labels):
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    acc = accuracy_score(labels, preds)
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
