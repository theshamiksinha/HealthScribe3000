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
from inference.evaluate_summariser import evaluate_pegasus_model
from inference.eval_perspective_wise import evaluate_perspective_wise

def train_llm():
    # Load config
    config = load_config()
    
    # Load data
    print("Loading training and validation data...")
    train_data = load_dataset(file_path=config['data']['train_path'])
    val_data = load_dataset(file_path=config['data']['val_path'])
    test_data = load_dataset(file_path=config['data']['test_path'])
    
    # For faster test runs (adjust/remove for real training)
    # train_data = train_data[:int(len(train_data) * 0.01)]
    # val_data = val_data[:int(len(val_data) * 0.05)]
    # test_data = test_data[:int(len(val_data) * 0.05)]
    
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
    
    print("LLM training completed!\n")
    
    # Save the trained model
    print("saving the trained model\n")
    # Ensure the directory exists
    os.makedirs(config["training"]["llm"]["save_dir"], exist_ok=True)
    trainer.save_model(config["training"]["llm"]["save_dir"])
    tokenizer.save_pretrained(config["training"]["llm"]["save_dir"])

    # ########################################################################################################### #
    """GENERATE SAMPLE PREDICTIONS"""
    # test_dataset = LLMDataset(test_data, tokenizer, config, mode="test")

    # evaluate_pegasus_model(model, tokenizer, test_dataset, output_dir="eval_after_training")
    # evaluate_perspective_wise(model, tokenizer, test_dataset, all_perspectives=list(config["perspectives"].keys()))
    
    # print("\nGenerating predictions for first 10 validation samples...")
    # model.eval()
    # for i in range(10):
    #     sample = val_dataset[i]
    #     input_ids = sample["input_ids"].unsqueeze(0).to(model.device)
    #     attention_mask = sample["attention_mask"].unsqueeze(0).to(model.device)

    #     with torch.no_grad():
    #         output_ids = model.generate(
    #             input_ids=input_ids,
    #             attention_mask=attention_mask,
    #             max_length=config['model']['llm']['max_length'],  # or 128/256 etc
    #             num_beams=4,
    #             early_stopping=True,
    #         )

    #     decoded_input = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    #     decoded_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    #     label_ids = sample["labels"].clone()
    #     label_ids[label_ids == -100] = tokenizer.pad_token_id
    #     decoded_reference = tokenizer.decode(label_ids, skip_special_tokens=True)


    #     print(f"\n--- Sample {i+1} ---")
    #     print(f"INPUT:\n{decoded_input}\n")
    #     print(f"PREDICTED:\n{decoded_output}\n")
    #     print(f"REFERENCE:\n{decoded_reference}\n")
        
    #     print("\n")

if __name__ == "__main__":
    train_llm()
