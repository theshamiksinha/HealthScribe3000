import json
import os
from typing import Any, Dict, List

import yaml


def load_dataset(file_path: str) -> List[Dict[str, Any]]:
    """Load dataset from JSON file"""
    with open(file_path, 'r') as f:
        return json.load(f)


def save_dataset(data, file_path: str) -> None:
    """Save dataset to JSON file"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)


def load_config(path: str = 'config.yaml') -> Any:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def save_predictions_to_json(test_data, output_path: str = "predicted_test_data.json") -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)
