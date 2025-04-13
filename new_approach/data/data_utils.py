# data/data_utils.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import json
import os
from tqdm import tqdm
import yaml

def load_dataset(file_path):
    """Load dataset from JSON file"""
    with open(file_path, 'r') as f:
        return json.load(f)

def save_dataset(data, file_path):
    """Save dataset to JSON file"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

def load_config(path='config/config.yaml'):
    with open(path, 'r') as f:
        return yaml.safe_load(f)
    
def prepare_llm_training_data(data, perspective_classifier, config):
    """
    Prepare data for LLM fine-tuning by adding perspective labels.
    
    Args:
        data: Raw data with questions and answers
        perspective_classifier: Trained perspective classifier model
        config: Configuration dictionary
        
    Returns:
        Processed data with perspective labels
    """
    processed_data = []
    
    for instance in tqdm(data, desc="Preparing LLM training data"):
        processed_instance = {
            "question": instance["question"],
            "answers": []
        }
        
        for answer in instance["answers"]:
            # Predict perspectives if not already labeled
            if "perspectives" not in answer:
                perspectives = perspective_classifier.predict(
                    instance["question"], 
                    answer["text"]
                )
                answer["perspectives"] = perspectives
            
            processed_instance["answers"].append({
                "text": answer["text"],
                "perspectives": answer["perspectives"]
            })
        
        # If labeled spans exist, use them for training output
        if "labelled_answer_spans" in instance:
            perspective_summaries = {}
            
            for perspective, spans in instance["labelled_answer_spans"].items():
                if spans:
                    # Join all spans for this perspective
                    span_texts = [span["txt"] for span in spans]
                    perspective_summaries[perspective] = " ".join(span_texts)
            
            processed_instance["perspective_summaries"] = perspective_summaries
        
        processed_data.append(processed_instance)
    
    return processed_data