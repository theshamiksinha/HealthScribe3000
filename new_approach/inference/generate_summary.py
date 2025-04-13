# inference/generate_summary.py
import os
import sys
import json
import yaml
import torch
from transformers import AutoTokenizer
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.llm_model import PerspectiveLLM
from models.perspective_classifier import PerspectiveClassifier
from data.data_utils import load_dataset, save_dataset

def load_config(path='config/config.yaml'):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def generate_summaries(input_file, output_file, model_path=None, classify_perspectives=True):
    """
    Generate perspective-based summaries for a dataset
    
    Args:
        input_file: Path to input JSON file with questions and answers
        output_file: Path to output JSON file for saving summaries
        model_path: Path to fine-tuned LLM model
        classify_perspectives: Whether to classify perspectives first
    """
    # Load config
    config = load_config()
    
    # Set default model path if not provided
    if model_path is None:
        model_path = os.path.join(config['training']['llm']['save_dir'], 'best_model')
    
    # Load data
    print(f"Loading data from {input_file}...")
    data = load_dataset(input_file)
    
    # Load models
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    llm = PerspectiveLLM(model_path, tokenizer=tokenizer)
    
    # Load perspective classifier if needed
    if classify_perspectives:
        print("Loading perspective classifier...")
        classifier = PerspectiveClassifier.from_pretrained(
            os.path.join(config['training']['classifier']['save_dir'], 'best_model.pt'),
            config['model']['classifier']['encoder_model']
        )
    
    # Process each question-answer pair
    results = []
    for instance in tqdm(data, desc="Generating summaries"):
        question = instance["question"]
        answers = [a["text"] for a in instance["answers"]]
        
        # First, classify perspectives if needed
        if classify_perspectives:
            # Get all unique perspectives across answers
            all_perspectives = set()
            for answer_text in answers:
                perspectives = classifier.predict(question, answer_text)
                all_perspectives.update(perspectives)
        else:
            # Use all defined perspectives
            all_perspectives = set(config['perspectives'].keys())
        
        # Generate summaries
        summaries = llm.generate(
            question=question,
            answers=answers,
            perspectives=list(all_perspectives),
            max_length=config['inference']['max_length']
        )
        
        # Store results
        result = {
            "question": question,
            "answers": instance["answers"],
            "perspectives": list(all_perspectives),
            "perspective_summaries": summaries
        }
        
        results.append(result)
    
    # Save results
    print(f"Saving results to {output_file}...")
    save_dataset(results, output_file)
    print(f"Generated summaries for {len(results)} questions.")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate perspective-based summaries")
    parser.add_argument("--input", required=True, help="Input JSON file path")
    parser.add_argument("--output", required=True, help="Output JSON file path")
    parser.add_argument("--model", default=None, help="Path to fine-tuned model")
    parser.add_argument("--no-classify", action="store_true", help="Skip perspective classification")
    
    args = parser.parse_args()
    
    generate_summaries(
        args.input,
        args.output,
        model_path=args.model,
        classify_perspectives=not args.no_classify
    )