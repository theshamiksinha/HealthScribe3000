from training.train_llm import train_llm
from transformers import PegasusTokenizer, PegasusForConditionalGeneration
from data.llm_dataset import LLMDataset
import torch
import os
from inference.evaluate_summariser import evaluate_pegasus_model
from inference.eval_perspective_wise import evaluate_perspective_wise

def train_or_load_summariser(config):
    model_dir = config["training"]["llm"]["save_dir"]

    # Train only if the fine-tuned model doesn't exist
    if not os.path.exists(model_dir):
        print(f"Fine-tuned model not found at {model_dir}. Training new model...")
        train_llm()

    print(f"Loading fine-tuned model from {model_dir}")
    model = PegasusForConditionalGeneration.from_pretrained(model_dir)
    tokenizer = PegasusTokenizer.from_pretrained(model_dir)
    
    model.eval()
    return model, tokenizer

def generate_summaries(model, tokenizer, test_data, config):
    test_dataset = LLMDataset(test_data, tokenizer, config, mode="test")
    
    evaluate_pegasus_model(model, tokenizer, test_dataset, output_dir="eval_after_training")
    evaluate_perspective_wise(model, tokenizer, test_dataset, all_perspectives=list(config["perspectives"].keys()))

    print("\nGenerating summaries on test set...")
    model.eval()
    device = next(model.parameters()).device  # Get model device

    for i in range(10):
        sample = test_dataset[i]
        input_ids = sample["input_ids"].unsqueeze(0).to(device)
        attention_mask = sample["attention_mask"].unsqueeze(0).to(device)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=config['model']['llm']['max_length'],
                num_beams=4,
                early_stopping=True,
            )

        input_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # Only decode reference summary if labels are available
        if "labels" in sample:
            labels = sample["labels"].to(device)
            ref_text = tokenizer.decode(
                labels.masked_fill(labels == -100, tokenizer.pad_token_id),
                skip_special_tokens=True,
            )
        else:
            ref_text = "[No reference summary available]"

        print(f"\n📝 INPUT:\n{input_text}\n")
        print(f"🔮 PREDICTED SUMMARY:\n{output_text}\n")
        print(f"✅ REFERENCE SUMMARY:\n{ref_text}\n")

