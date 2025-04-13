# Load the dataset
# Initialize SpanExtractorWithCRF
# Set up optimizer, loss, and metrics
# Run the training loop
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Train the SpanExtractorWithCRF model
import sys
import os
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
    train_data = train_data[:int(len(train_data) * 0.01)]
    val_data = val_data[:int(len(val_data) * 0.01)]

    
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
        evaluate(model, val_loader, tokenizer, label_map_reverse, device)
        
        
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
    
    Args:
        model: The trained SpanExtractorWithCRF model
        val_loader: DataLoader for validation data
        tokenizer: Tokenizer used for encoding
        id2label: Mapping from tag IDs to tag names
        device: Device to run inference on
        num_examples: Number of examples to print
    
    Returns:
        Dictionary with validation metrics
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    # For printing examples
    examples_to_print = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['labels'].to(device)
            
            # Get model predictions
            preds = model(input_ids, attention_mask, token_type_ids)
            
            # Collect predictions and labels
            for i, pred_seq in enumerate(preds):
                # Get the actual sequence length (non-padding)
                seq_len = attention_mask[i].sum().item()
                
                # Get the predicted and gold labels
                pred_tags = [id2label[tag_id] for tag_id in pred_seq[:seq_len]]
                true_tags = [id2label[tag_id] for tag_id in labels[i][:seq_len].cpu().tolist()]
                
                all_preds.append(pred_seq[:seq_len])
                all_labels.append(labels[i][:seq_len].cpu().tolist())
                
                # Store examples for later printing
                if len(examples_to_print) < num_examples and batch_idx % (len(val_loader) // num_examples) == 0:
                    # Decode the input tokens
                    tokens = tokenizer.convert_ids_to_tokens(input_ids[i][:seq_len].cpu().tolist())
                    
                    # Find where the answer part starts (based on token_type_ids)
                    answer_start = 0
                    for j in range(seq_len):
                        if token_type_ids[i][j] == 1:
                            answer_start = j
                            break
                            
                    # Only keep tokens from the answer
                    answer_tokens = tokens[answer_start:]
                    answer_pred_tags = pred_tags[answer_start:]
                    answer_true_tags = true_tags[answer_start:]
                    
                    examples_to_print.append({
                        'tokens': answer_tokens,
                        'pred_tags': answer_pred_tags,
                        'true_tags': answer_true_tags
                    })
    
    # Calculate F1 score
    f1, report = compute_token_f1(all_preds, all_labels, id2label)
    
    # Print examples
    print("\n===== PREDICTION EXAMPLES =====")
    for idx, example in enumerate(examples_to_print):
        print(f"\nExample {idx+1}:")
        print("Original Text with Gold Spans:")
        _print_tagged_text(example['tokens'], example['true_tags'])
        
        print("\nPredicted Spans:")
        _print_tagged_text(example['tokens'], example['pred_tags'])
        print("-" * 80)
    
    return {
        'f1': f1,
        'report': report
    }

def _print_tagged_text(tokens, tags):
    """Helper function to print text with colored spans for different perspective types."""
    # Terminal colors
    colors = {
        'B-INFORMATION': '\033[94m',  # Blue
        'I-INFORMATION': '\033[94m',
        'B-SUGGESTION': '\033[92m',   # Green
        'I-SUGGESTION': '\033[92m',
        'B-CAUSE': '\033[91m',        # Red
        'I-CAUSE': '\033[91m',
        'B-EXPERIENCE': '\033[93m',   # Yellow
        'I-EXPERIENCE': '\033[93m',
        'B-QUESTION': '\033[95m',     # Magenta
        'I-QUESTION': '\033[95m',
        'O': '\033[0m',               # Reset
        'END': '\033[0m'              # Reset
    }
    
    # Format special tokens to be more readable
    formatted_tokens = []
    for token in tokens:
        if token.startswith('##'):
            formatted_tokens.append(token[2:])
        elif token in ['[CLS]', '[SEP]', '[PAD]']:
            continue
        else:
            formatted_tokens.append(' ' + token)
    
    # Print the text with colors
    current_perspective = None
    output_text = ""
    
    for token, tag in zip(formatted_tokens, tags):
        if tag.startswith('B-'):
            # Start of a new perspective span
            if current_perspective:
                output_text += colors['END']
            current_perspective = tag[2:]
            output_text += colors[tag] + token
        elif tag.startswith('I-'):
            # Continuation of a perspective span
            output_text += token
        else:  # 'O' tag
            if current_perspective:
                output_text += colors['END']
                current_perspective = None
            output_text += token
    
    # End any open color codes
    if current_perspective:
        output_text += colors['END']
    
    print(output_text)


if __name__ == "__main__":
    train()