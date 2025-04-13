# utils/metrics.py

from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics import f1_score, precision_score, recall_score

def compute_token_f1(preds, labels, label_map):
    """
    Compute token-level F1 score between predicted and ground truth label sequences.
    
    Arguments:
        preds: List of predicted label ids (e.g., [[0, 1, 2], [1, 0, 2], ...])
        labels: List of true label ids (e.g., [[0, 1, 2], [1, 0, 2], ...])
        label_map: A dictionary mapping label ids to tag strings 
                   (e.g., {0: 'O', 1: 'B-INFORMATION', 2: 'I-INFORMATION'})

    Returns:
        f1: The computed F1 score for token-level predictions using seqeval.
    """
    # Convert label ids to tag strings using the label_map
    preds_str = [[label_map[i] for i in seq] for seq in preds]
    labels_str = [[label_map[i] for i in seq] for seq in labels]

    # Now filter out padding tokens (usually 'O' or a special tag like '<PAD>')
    # If padding is included in your label map, you should remove it by checking a pad_token_label_id (like -100)
    # Assuming 0 corresponds to "O", we usually do NOT remove it, since "O" is a valid tag

    # Compute F1 score using seqeval
    f1 = f1_score(labels_str, preds_str)
    return f1


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
