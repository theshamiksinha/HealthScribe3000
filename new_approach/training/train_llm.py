# training/train_llm.py
import os
import sys
import yaml
import json
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.llm_dataset import LLMDataset
from data.data_utils import load_dataset, prepare_llm_training_data
from models.perspective_classifier import PerspectiveClassifier
from utils.metrics import compute_rouge

def load_config(path='config/config.yaml'):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def train_llm():
    # Load config
    config = load_config()
    
    # Load data
    print("Loading datasets...")
    train_data = load_dataset(config['data']['train_path'])
    val_data = load_dataset(config['data']['val_path'])
    
    # Load perspective classifier for labeling
    if any("perspectives" not in answer for instance in train_data for answer in instance["answers"]):
        print("Loading perspective classifier...")
        classifier = PerspectiveClassifier.from_pretrained(
            os.path.join(config['training']['classifier']['save_dir'], 'best_model.pt'),
            config['model']['classifier']['encoder_model']
        )
        
        # Prepare data with perspective labels
        print("Preparing data with perspective labels...")
        train_data = prepare_llm_training_data(train_data, classifier, config)
        val_data = prepare_llm_training_data(val_data, classifier, config)
    
    # Initialize tokenizer and model
    print("Initializing model and tokenizer...")
    model_name = config['model']['llm']['base_model']
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    # Create datasets
    train_dataset = LLMDataset(train_data, tokenizer, config, mode="train")
    val_dataset = LLMDataset(val_data, tokenizer, config, mode="val")
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding='longest'
    )
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=config['training']['llm']['save_dir'],
        num_train_epochs=config['training']['llm']['num_epochs'],
        per_device_train_batch_size=config['training']['llm']['batch_size'],
        per_device_eval_batch_size=config['training']['llm']['batch_size'],
        gradient_accumulation_steps=config['training']['llm']['gradient_accumulation_steps'],
        learning_rate=config['training']['llm']['learning_rate'],
        weight_decay=0.01,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        logging_dir=os.path.join(config['training']['llm']['save_dir'], 'logs'),
        load_best_model_at_end=True,
        metric_for_best_model="eval_rouge2",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )
    
    # Define compute metrics function
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
        
        # Decode predictions and references
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Compute ROUGE scores
        rouge_scores = compute_rouge(decoded_preds, decoded_labels)
        
        return rouge_scores
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    # Train the model
    print("Training LLM...")
    trainer.train()
    
    # Save the best model
    print("Saving the best model...")
    trainer.save_model(os.path.join(config['training']['llm']['save_dir'], 'best_model'))
    
    # Save the configuration with the model
    with open(os.path.join(config['training']['llm']['save_dir'], 'best_model', 'config.yaml'), 'w') as f:
        yaml.dump(config, f)
    
    print("LLM training completed!")

if __name__ == "__main__":
    train_llm()