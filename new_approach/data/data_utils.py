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

def save_predictions_to_json(test_data, output_path="predicted_test_data.json"):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)