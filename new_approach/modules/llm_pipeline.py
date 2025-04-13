from training.train_llm import train_llm
from transformers import PegasusTokenizer, PegasusForConditionalGeneration
from data.llm_dataset import LLMDataset
import torch
import os

def train_or_load_summariser(config):
    model_dir = config["training"]["llm"]["save_dir"]

    # Train only if the fine-tuned model doesn't exist
    if not os.path.exists(model_dir):
        print(f"Fine-tuned model not found at {model_dir}. Training new model...")
        train_llm()

    print(f"Loading fine-tuned model from {model_dir}")
    model = PegasusForConditionalGeneration.from_pretrained(model_dir)
    tokenizer = PegasusTokenizer.from_pretrained(model_dir)
    return model, tokenizer

def generate_summaries(model, tokenizer, test_data, config):
    test_dataset = LLMDataset(test_data, tokenizer, config, mode="test")

    print("\nGenerating summaries on test set...")
    model.eval()
    for i in range(10):
        sample = test_dataset[i]
        input_ids = sample["input_ids"].unsqueeze(0)
        attention_mask = sample["attention_mask"].unsqueeze(0)

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
        ref_text = tokenizer.decode(
            sample["labels"].masked_fill(sample["labels"] == -100, tokenizer.pad_token_id),
            skip_special_tokens=True,
        )
        print(f"\n📝 INPUT:\n{input_text}\n")
        print(f"🔮 PREDICTED SUMMARY:\n{output_text}\n")
        print(f"✅ REFERENCE SUMMARY:\n{ref_text}\n")
