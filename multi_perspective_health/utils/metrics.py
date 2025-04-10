# utils/metrics.py

from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_token_f1(true_labels, pred_labels):
    """
    Computes precision, recall, and F1 score for token-level BIO tagging.
    
    Args:
        true_labels (List[List[str]]): True BIO labels for each sentence.
        pred_labels (List[List[str]]): Predicted BIO labels for each sentence.
    
    Returns:
        Dict[str, float]: Precision, recall, F1-score.
    """
    assert len(true_labels) == len(pred_labels), "Mismatch in number of sequences"
    
    return {
        "precision": precision_score(true_labels, pred_labels),
        "recall": recall_score(true_labels, pred_labels),
        "f1": f1_score(true_labels, pred_labels)
    }

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
