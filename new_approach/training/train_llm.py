import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import os
import yaml
import json
import torch
from tqdm import tqdm
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)
from data.data_utils import load_dataset, load_config
import numpy as np
from data.llm_dataset import LLMDataset
from utils.metrics import compute_rouge
from transformers import PegasusTokenizer, PegasusForConditionalGeneration

def train_llm():
    # Load config
    config = load_config()
    
    # Load data
    print("Loading training and validation data...")
    train_data = load_dataset(config['data']['train_path'])
    val_data = load_dataset(config['data']['val_path'])

    # # Initialize tokenizer and model
    # model_name = config['model']['llm']['base_model']
    # tokenizer = AutoTokenizer.from_pretrained(model_name)

    # # Make sure pad_token is properly set for Pegasus
    # if tokenizer.pad_token is None:
    #     tokenizer.pad_token = tokenizer.eos_token

    # # Ensure decoder_start_token_id is set properly
    # model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    # model.config.decoder_start_token_id = tokenizer.pad_token_id
    

    tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-xsum")  # or your variant
    model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-xsum")


    # Create datasets with tqdm during preprocessing
    print("Tokenizing and creating training dataset...")
    train_dataset = LLMDataset(
        list(tqdm(train_data, desc="Processing train data")), 
        tokenizer, config, mode="train"
    )

    print("Tokenizing and creating validation dataset...")
    val_dataset = LLMDataset(
        list(tqdm(val_data, desc="Processing val data")), 
        tokenizer, config, mode="val"
    )

    # Data collator
    data_collator= DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100
    )
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=config['training']['llm']['save_dir'],
        num_train_epochs=config['training']['llm']['num_epochs'],
        per_device_train_batch_size=config['training']['llm']['batch_size'],
        per_device_eval_batch_size=config['training']['llm']['batch_size'],
        gradient_accumulation_steps=config['training']['llm']['gradient_accumulation_steps'],
        learning_rate=float(config['training']['llm']['learning_rate']),
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
        report_to="none",
        data_collator = data_collator,
    )
    
    # Define compute metrics function
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
        
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
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
    
    print("Training LLM...")
    trainer.train()
    
    print("LLM training completed!")

if __name__ == "__main__":
    train_llm()
