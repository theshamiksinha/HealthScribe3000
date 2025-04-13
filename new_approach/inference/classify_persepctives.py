# inference/classify_perspectives.py
import os
import sys
import json
import yaml
import torch
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.perspective_classifier import PerspectiveClassifier
from data.data_utils import load_dataset, save_dataset

def load_config(path='config/config.yaml'):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def classify_perspectives(input_file, output_file, model_path=None):
    """
    Classify perspectives in health-related answers
    
    Args:
        input_file: Path to input JSON file with questions and answers
        output_file: Path to output JSON file for saving classified data
        model_path: Path to trained classifier model
    """
    # Load config
    config = load_config()
    
    # Set default model path if not provided
    if model_path is None:
        model_path = os.path.join(config['training']['classifier']['save_dir'], 'best_model.pt')
    
    # Load data
    print(f"Loading data from {input_file}...")
    data = load_dataset(input_file)
    
    # Load perspective classifier
    print(f"Loading perspective classifier from {model_path}...")
    classifier = PerspectiveClassifier.from_pretrained(
        model_path,
        config['model']['classifier']['encoder_model']
    )
    
    # Process each question-answer pair
    results = []
    for instance in tqdm(data, desc="Classifying perspectives"):
        question = instance["question"]
        
        # Process each answer
        processed_answers = []
        for answer in instance["answers"]:
            answer_text = answer["text"]
            
            # Classify perspectives
            perspectives = classifier.predict(question, answer_text)
            
            # Store result
            processed_answer = {
                "text": answer_text,
                "perspectives": perspectives
            }
            
            # Keep any existing fields
            for key, value in answer.items():
                if key not in ["text", "perspectives"]:
                    processed_answer[key] = value
            
            processed_answers.append(processed_answer)
        
        # Create result instance
        result = {
            "question": question,
            "answers": processed_answers
        }
        
        # Keep any other fields from the original instance
        for key, value in instance.items():
            if key not in ["question", "answers"]:
                result[key] = value
        
        results.append(result)
    
    # Save results
    print(f"Saving classified data to {output_file}...")
    save_dataset(results, output_file)
    print(f"Classified perspectives for {len(results)} questions.")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Classify perspectives in health answers")
    parser.add_argument("--input", required=True, help="Input JSON file path")
    parser.add_argument("--output", required=True, help="Output JSON file path")
    parser.add_argument("--model", default=None, help="Path to trained classifier model")
    
    args = parser.parse_args()
    
    classify_perspectives(
        args.input,
        args.output,
        model_path=args.model
    )