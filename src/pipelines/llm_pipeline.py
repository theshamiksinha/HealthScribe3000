import os
from typing import Tuple, Dict, List

import torch
from transformers import PegasusTokenizer, PegasusForConditionalGeneration

from data.llm_dataset import LLMDataset
from training.train_llm import train_llm


def train_or_load_summariser(config: Dict) -> Tuple[PegasusForConditionalGeneration, PegasusTokenizer]:
    model_dir = config["training"]["llm"]["save_dir"]

    if not os.path.exists(model_dir):
        print(f"Fine-tuned model not found at {model_dir}. Training new model...")
        train_llm()

    print(f"Loading fine-tuned model from {model_dir}")
    model = PegasusForConditionalGeneration.from_pretrained(model_dir)
    tokenizer = PegasusTokenizer.from_pretrained(model_dir)

    model.eval()
    return model, tokenizer


def generate_summaries(model: PegasusForConditionalGeneration, tokenizer: PegasusTokenizer, test_data: List,
                       config: Dict) -> None:
    test_dataset = LLMDataset(test_data, tokenizer, config, mode="test")

    # uncomment for evaluation on metrics
    # evaluate_pegasus_model(model, tokenizer, test_dataset, output_dir="eval_after_training")
    # evaluate_perspective_wise(model, tokenizer, test_dataset, all_perspectives=list(config["perspectives"].keys()))

    print("\nGenerating summaries on test set...")
    model.eval()
    device = next(model.parameters()).device

    for sample in test_dataset:
        input_ids = sample["input_ids"].unsqueeze(0).to(device)
        attention_mask = sample["attention_mask"].unsqueeze(0).to(device)

        perspective = sample["perspective"]
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

        output_text = f"{perspective}_SUMMARY: " + output_text.split(":", 1)[-1].strip()

        # Only decode reference summary if labels are available
        if "labels" in sample:
            labels = sample["labels"].to(device)
            ref_text = tokenizer.decode(
                labels.masked_fill(labels == -100, tokenizer.pad_token_id),
                skip_special_tokens=True,
            )
        else:
            ref_text = "[No reference summary available]"

        print(f"\nINPUT:\n{input_text}\n")
        print(f"PREDICTED SUMMARY:\n{output_text}\n")
