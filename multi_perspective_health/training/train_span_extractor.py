# Load the dataset
# Initialize SpanExtractorWithCRF
# Set up optimizer, loss, and metrics
# Run the training loop
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Train the SpanExtractorWithCRF model
import numpy as np
import json
import yaml
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from models.span_extractor import SpanExtractorWithCRF
from data.span_dataset import SpanDataset
from utils.metrics import compute_token_f1
from tqdm import tqdm


def load_config(path='config/config.yaml'):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def train():
    config = load_config()
    model_name = config['model']['encoder_model']
    batch_size = config['training']['batch_size']
    epochs = config['training']['num_epochs']
    lr = float(config['training']['learning_rate'])
    
    # Add default values if not in config
    num_tags = config.get('model', {}).get('num_tags', 11)  # For 5 perspectives (B+I) + O
    
    # Load data
    with open(config['data']['train_path'], 'r') as f:
        train_data = json.load(f) 
    with open(config['data']['val_path'], 'r') as f:
        val_data = json.load(f)
        
    # For faster test runs (adjust/remove for real training)
    train_data = train_data[:int(len(train_data) * 0.1)]
    val_data = val_data[:int(len(val_data) * 0.1)]

    
    # Create label map if not provided
    if 'label_map' not in config:
        perspectives = ["INFORMATION", "SUGGESTION", "CAUSE", "EXPERIENCE", "QUESTION"]
        label_map = {'O': 0}
        idx = 1
        for perspective in perspectives:
            label_map[f'B-{perspective}'] = idx
            idx += 1
            label_map[f'I-{perspective}'] = idx
            idx += 1
        config['label_map'] = label_map
    else:
        label_map = config['label_map']
    
    # Create reverse label map
    label_map_reverse = {v: k for k, v in label_map.items()}
    config['label_map_reverse'] = label_map_reverse
    
    # Initialize tokenizer and datasets
    tokenizer = AutoTokenizer.from_pretrained(config['data']['tokenizer_name'])
    train_dataset = SpanDataset(train_data, tokenizer, label_map)
    val_dataset = SpanDataset(val_data, tokenizer, label_map)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = SpanExtractorWithCRF(model_name, num_tags).to(device)
    
    # Set up optimizer with weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=lr,
        weight_decay=0.01
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    
    best_f1 = 0.0
    patience = 0
    max_patience = 3  # Early stopping after 3 epochs without improvement
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        total_loss = 0
        train_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training")
        
        for batch in train_progress:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass and loss calculation
            loss = model(input_ids, attention_mask, token_type_ids, labels)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Update progress
            total_loss += loss.item()
            train_progress.set_postfix({'loss': total_loss / (train_progress.n + 1)})
        
        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}")
        
        # Validation phase
        model.eval()
        all_preds = []
        all_labels = []
        total_val_loss = 0
        
        with torch.no_grad():
            val_progress = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} - Validation")
            
            for batch in val_progress:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                token_type_ids = batch['token_type_ids'].to(device)
                labels = batch['labels'].to(device)
                
                # For validation loss
                loss = model(input_ids, attention_mask, token_type_ids, labels)
                total_val_loss += loss.item()
                
                # For predictions
                preds = model(input_ids, attention_mask, token_type_ids)
                
                # Collect predictions and labels
                for i, pred_seq in enumerate(preds):
                    # Truncate predictions to actual sequence length (non-padding)
                    seq_len = attention_mask[i].sum().item()
                    all_preds.append(pred_seq[:seq_len])
                    
                    # Get the actual labels (also truncated)
                    actual_labels = labels[i][:seq_len].cpu().tolist()
                    all_labels.append(actual_labels)
        
        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Validation Loss: {avg_val_loss:.4f}")
        
        # Calculate F1 score
        f1, report = compute_token_f1(all_preds, all_labels, label_map_reverse)
        print(f"Validation F1: {f1:.4f}")
        print(report)
        
        # Perform detailed evaluation with examples
        print("\nDetailed Evaluation with Examples:")
        model.eval()
        print(f"\nEvaluating model after epoch {epoch+1}...")
        report, entity_metrics = evaluate(model, val_loader, tokenizer, label_map_reverse, device)

        # Calculate overall F1 for model saving
        f1 = np.mean([metrics['f1'] for metrics in entity_metrics.values()])
                
        
        # Save best model
        # if f1 > best_f1:
        #     best_f1 = f1
        #     torch.save({
        #         'epoch': epoch,
        #         'model_state_dict': model.state_dict(),
        #         'optimizer_state_dict': optimizer.state_dict(),
        #         'f1': f1,
        #     }, os.path.join(config['training']['save_dir'], 'best_span_model.pt'))
        #     print(f"✅ New best model saved with F1: {f1:.4f}")
        #     patience = 0
        # else:
        #     patience += 1
        #     if patience >= max_patience:
        #         print(f"Early stopping after {patience} epochs without improvement")
        #         break
        
        # Update learning rate
        
        scheduler.step()

    print(f"Training completed. Best F1: {best_f1:.4f}")
    
 
def evaluate(model, val_loader, tokenizer, id2label, device, num_examples=5):
    """
    Evaluate the model and print examples of predictions vs. ground truth.
    Focus on examples where the model both succeeds and fails at detecting spans.
    
    Args:
        model: The trained SpanExtractorWithCRF model
        val_loader: DataLoader for validation data
        tokenizer: Tokenizer used for encoding
        id2label: Mapping from tag IDs to tag names
        device: Device to run inference on
        num_examples: Number of examples to print
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    # For storing all examples with their predictions
    all_examples = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['labels'].to(device)
            
            # Get model predictions
            preds = model(input_ids, attention_mask, token_type_ids)
            
            # Collect predictions and labels for each sequence
            for i, pred_seq in enumerate(preds):
                seq_len = attention_mask[i].sum().item()
                pred_tags = [id2label[tag_id] for tag_id in pred_seq[:seq_len]]
                true_tags = [id2label[tag_id] for tag_id in labels[i][:seq_len].cpu().tolist()]
                
                all_preds.extend(pred_seq[:seq_len])
                all_labels.extend(labels[i][:seq_len].cpu().tolist())
                
                # Find where the answer part starts (based on token_type_ids)
                answer_start = 0
                for j in range(seq_len):
                    if token_type_ids[i][j].item() == 1:
                        answer_start = j
                        break
                
                # Only keep tokens from the answer
                tokens = tokenizer.convert_ids_to_tokens(input_ids[i][:seq_len].cpu().tolist())
                answer_tokens = tokens[answer_start:seq_len]
                answer_pred_tags = pred_tags[answer_start:seq_len]
                answer_true_tags = true_tags[answer_start:seq_len]
                
                # Count how many non-O tags in gold and predictions
                gold_spans = sum(1 for tag in answer_true_tags if tag != 'O')
                pred_spans = sum(1 for tag in answer_pred_tags if tag != 'O')
                
                # Only store examples with at least one gold span
                if gold_spans > 0:
                    all_examples.append({
                        'tokens': answer_tokens,
                        'pred_tags': answer_pred_tags,
                        'true_tags': answer_true_tags,
                        'gold_spans': gold_spans,
                        'pred_spans': pred_spans,
                        # Calculate an "interestingness" score - higher when predictions differ from gold
                        'interestingness': sum(1 for t, p in zip(answer_true_tags, answer_pred_tags) if t != p) / len(answer_true_tags)
                    })
    
    # Calculate F1 scores for each perspective type
    perspectives = ["INFORMATION", "SUGGESTION", "CAUSE", "EXPERIENCE", "QUESTION"]
    
    # Convert flat lists to tag names
    y_true = [id2label[label] if isinstance(label, int) else label for label in all_labels]
    y_pred = [id2label[pred] if isinstance(pred, int) else pred for pred in all_preds]
    
    # Print detailed performance by perspective
    print("\n===== PERFORMANCE BY PERSPECTIVE =====")
    report = classification_report(y_true, y_pred)
    print(report)
    
    # Calculate entity-level F1 scores (not just token-level)
    entity_metrics = calculate_entity_level_f1(all_examples)
    print("\n===== ENTITY-LEVEL METRICS =====")
    print("Perspective\tPrecision\tRecall\tF1")
    for persp, metrics in entity_metrics.items():
        print(f"{persp}\t{metrics['precision']:.3f}\t{metrics['recall']:.3f}\t{metrics['f1']:.3f}")
    
    # Find good and bad examples to print
    sorted_examples = sorted(all_examples, key=lambda x: x['interestingness'], reverse=True)
    
    # Select a mix of interesting examples
    examples_to_print = []
    # First, add some examples where the model completely missed spans
    missed_examples = [ex for ex in sorted_examples if ex['gold_spans'] > 0 and ex['pred_spans'] == 0]
    if missed_examples:
        examples_to_print.extend(missed_examples[:min(2, len(missed_examples))])
    
    # Add some examples where the model found some spans but missed others
    partial_examples = [ex for ex in sorted_examples 
                       if ex['gold_spans'] > ex['pred_spans'] > 0 
                       and ex not in examples_to_print]
    if partial_examples:
        examples_to_print.extend(partial_examples[:min(2, len(partial_examples))])
    
    # Add some examples where the model did well
    good_examples = [ex for ex in sorted_examples 
                    if ex['gold_spans'] > 0 and ex['interestingness'] < 0.3 
                    and ex not in examples_to_print]
    if good_examples:
        examples_to_print.extend(good_examples[:min(1, len(good_examples))])
    
    # Print examples
    print("\n===== PREDICTION EXAMPLES =====")
    for idx, example in enumerate(examples_to_print[:num_examples]):
        print(f"\nExample {idx+1}:")
        print("Original Text with Gold Spans:")
        print_tagged_text(example['tokens'], example['true_tags'])
        
        print("\nPredicted Spans:")
        print_tagged_text(example['tokens'], example['pred_tags'])
        print("-" * 80)
    
    return report, entity_metrics

def print_tagged_text(tokens, tags):
    """Print text with colored or bracketed spans for different perspective types."""
    # For terminals that support colors:
    use_colors = True  # Set to False if your terminal doesn't support colors
    
    colors = {
        'INFORMATION': ('\033[94m', '\033[0m'),  # Blue
        'SUGGESTION': ('\033[92m', '\033[0m'),   # Green
        'CAUSE': ('\033[91m', '\033[0m'),        # Red
        'EXPERIENCE': ('\033[93m', '\033[0m'),   # Yellow
        'QUESTION': ('\033[95m', '\033[0m'),     # Magenta
    }
    
    brackets = {
        'INFORMATION': ('[INFO]', '[/INFO]'),
        'SUGGESTION': ('[SUGG]', '[/SUGG]'),
        'CAUSE': ('[CAUSE]', '[/CAUSE]'),
        'EXPERIENCE': ('[EXP]', '[/EXP]'),
        'QUESTION': ('[Q]', '[/Q]'),
    }
    
    # Format special tokens for better readability
    text = ""
    current_tag = None
    
    for token, tag in zip(tokens, tags):
        # Handle special tokens
        if token in ['[CLS]', '[SEP]', '[PAD]']:
            continue
        
        # Format subword tokens
        if token.startswith('##'):
            token = token[2:]
        else:
            token = ' ' + token if text else token
        
        # Check for tag changes
        if tag.startswith('B-'):
            # Close previous tag if any
            if current_tag:
                text += colors[current_tag][1] if use_colors else brackets[current_tag][1]
            
            # Start new tag
            current_tag = tag[2:]  # Remove 'B-' prefix
            text += (colors[current_tag][0] if use_colors else brackets[current_tag][0]) + token
        
        elif tag.startswith('I-'):
            perspective = tag[2:]  # Remove 'I-' prefix
            
            # Handle case where an I- tag appears without a preceding B- tag
            if current_tag is None:
                current_tag = perspective
                text += (colors[current_tag][0] if use_colors else brackets[current_tag][0]) + token
            elif current_tag != perspective:
                # Tag changed without a B- tag (unusual but handle it)
                text += colors[current_tag][1] if use_colors else brackets[current_tag][1]
                current_tag = perspective
                text += (colors[current_tag][0] if use_colors else brackets[current_tag][0]) + token
            else:
                # Continue current tag
                text += token
        
        elif tag == 'O':
            # End any current tag
            if current_tag:
                text += colors[current_tag][1] if use_colors else brackets[current_tag][1]
                current_tag = None
            text += token
    
    # Close any open tag
    if current_tag:
        text += colors[current_tag][1] if use_colors else brackets[current_tag][1]
    
    print(text)

def calculate_entity_level_f1(examples):
    """
    Calculate entity-level (span-level) metrics instead of token-level.
    This is more useful for evaluating span extraction performance.
    """
    # Initialize counts for each perspective
    perspectives = ["INFORMATION", "SUGGESTION", "CAUSE", "EXPERIENCE", "QUESTION"]
    metrics = {p: {'tp': 0, 'fp': 0, 'fn': 0} for p in perspectives}
    
    for example in examples:
        tokens = example['tokens']
        true_tags = example['true_tags']
        pred_tags = example['pred_tags']
        
        # Extract spans from tags
        true_spans = extract_spans(tokens, true_tags)
        pred_spans = extract_spans(tokens, pred_tags)
        
        # Count true positives, false positives, and false negatives
        for perspective in perspectives:
            true_persp_spans = true_spans.get(perspective, [])
            pred_persp_spans = pred_spans.get(perspective, [])
            
            # Find matching spans (true positives)
            tp = 0
            for pred_span in pred_persp_spans:
                if any(overlap(pred_span, true_span) > 0.5 for true_span in true_persp_spans):
                    tp += 1
            
            # Calculate metrics
            fp = len(pred_persp_spans) - tp
            fn = len(true_persp_spans) - tp
            
            # Update counts
            metrics[perspective]['tp'] += tp
            metrics[perspective]['fp'] += fp
            metrics[perspective]['fn'] += fn
    
    # Calculate precision, recall, and F1 for each perspective
    results = {}
    for perspective, counts in metrics.items():
        tp = counts['tp']
        fp = counts['fp']
        fn = counts['fn']
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        results[perspective] = {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    return results

def extract_spans(tokens, tags):
    """Extract spans from BIO tags."""
    spans = {}
    current_span = None
    current_type = None
    
    for i, (token, tag) in enumerate(zip(tokens, tags)):
        if tag.startswith('B-'):
            # End previous span if any
            if current_span is not None:
                if current_type not in spans:
                    spans[current_type] = []
                spans[current_type].append((current_span, ' '.join(tokens[start:i])))
            
            # Start new span
            current_type = tag[2:]  # Remove 'B-' prefix
            start = i
            current_span = (start, i)
        
        elif tag.startswith('I-'):
            perspective = tag[2:]  # Remove 'I-' prefix
            
            # Continue current span or start a new one if none exists
            if current_span is not None and current_type == perspective:
                current_span = (current_span[0], i)
            elif current_type is None or current_type != perspective:
                # Handle cases where I- appears without B-
                current_type = perspective
                start = i
                current_span = (start, i)
        
        elif tag == 'O':
            # End previous span if any
            if current_span is not None:
                if current_type not in spans:
                    spans[current_type] = []
                spans[current_type].append((current_span, ' '.join(tokens[start:i])))
                current_span = None
                current_type = None
    
    # Handle span at the end of the sequence
    if current_span is not None:
        if current_type not in spans:
            spans[current_type] = []
        spans[current_type].append((current_span, ' '.join(tokens[start:len(tokens)])))
    
    return spans

def overlap(span1, span2):
    """Calculate the overlap between two spans."""
    s1, e1 = span1[0]
    s2, e2 = span2[0]
    
    overlap_start = max(s1, s2)
    overlap_end = min(e1, e2)
    
    if overlap_end < overlap_start:
        return 0
    
    overlap_length = overlap_end - overlap_start + 1
    span1_length = e1 - s1 + 1
    span2_length = e2 - s2 + 1
    
    return overlap_length / max(span1_length, span2_length)

if __name__ == "__main__":
    train()