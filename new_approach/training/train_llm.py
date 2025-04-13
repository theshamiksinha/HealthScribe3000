import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import os
import yaml
import json
import torch
from tqdm import tqdm
from transformers import (
    PegasusTokenizer,
    PegasusForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)

from data.data_utils import load_dataset, load_config
import numpy as np
from data.llm_dataset import LLMDataset
from utils.metrics import compute_rouge

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
    
    # For faster test runs (adjust/remove for real training)
    train_data = train_data[:int(len(train_data) * 0.1)]
    val_data = val_data[:int(len(val_data) * 0.1)]
    

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
    training_args = Seq2SeqTrainingArguments(
        output_dir="./pegasus_outputs",
        # evaluation_strategy="steps",
        eval_steps=100,
        # save_steps=100,
        save_strategy="no",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=3,
        fp16=True,  # if you're on GPU with mixed precision support
        report_to="none",
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
    trainer = Seq2SeqTrainer(
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
