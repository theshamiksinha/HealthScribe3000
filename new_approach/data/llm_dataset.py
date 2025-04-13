# data/llm_dataset.py
import json
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import numpy as np

class LLMDataset(Dataset):
    def __init__(self, data, tokenizer, config, mode="train"):
        """
        Dataset for LLM fine-tuning with perspective-based prompts
        
        Args:
            data: List of data instances with questions, answers, and perspective labels
            tokenizer: Tokenizer for the LLM
            config: Configuration dictionary
            mode: 'train', 'val', or 'test'
        """
        self.data = data
        self.tokenizer = tokenizer
        self.config = config
        self.mode = mode
        self.max_length = config['model']['llm']['max_length']
        self.perspectives = config['perspectives']
        self.examples = self.preprocess()
        
    def preprocess(self):
        """Convert raw data into prompted examples for LLM fine-tuning"""
        examples = []
        
        for instance in self.data:
            question = instance["question"]
            answers = instance["answers"]
            
            # Group answers by perspectives
            perspective_answers = {p: [] for p in self.perspectives.keys()}
            
            for answer in answers:
                # Get assigned perspectives (this assumes perspectives are predicted or labeled)
                perspectives = answer.get("perspectives", [])
                
                # Add the answer to each relevant perspective group
                for perspective in perspectives:
                    if perspective in self.perspectives:
                        perspective_answers[perspective].append(answer["text"])
            
            # Create input-output pairs for training
            input_prompt = self._create_input_prompt(question, perspective_answers)
            
            # Create expected output (for training)
            # In a real scenario, these would be the reference summaries for each perspective
            target_output = instance.get("perspective_summaries", self._create_default_output(perspective_answers))
            
            # Format target as a string
            target_text = self._format_target_output(target_output)
            
            # Tokenize inputs and targets
            inputs = self.tokenizer(
                input_prompt, 
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            targets = self.tokenizer(
                target_text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            example = {
                "input_ids": inputs["input_ids"][0],
                "attention_mask": inputs["attention_mask"][0],
                "labels": targets["input_ids"][0],
                "question": question,
                "perspective_answers": perspective_answers
            }
            
            examples.append(example)
        
        return examples
    
    def _create_input_prompt(self, question, perspective_answers):
        """Create a detailed prompt for the LLM"""
        prompt = f"Question: {question}\n\n"
        
        # Add perspective definitions
        prompt += "TASK: For each of the following perspectives, identify relevant spans in the provided answers and summarize them:\n\n"
        
        for perspective, definition in self.perspectives.items():
            prompt += f"- {perspective}: {definition['definition']}\n"
            prompt += f"  Tone: {definition['tone']}\n"
        
        prompt += "\nAnswers with their perspectives:\n"
        
        # Add perspective-specific answers
        for perspective, answers in perspective_answers.items():
            if answers:
                prompt += f"\n{perspective} ANSWERS:\n"
                for i, answer in enumerate(answers):
                    prompt += f"[{i+1}] {answer}\n"
        
        return prompt
    
    def _create_default_output(self, perspective_answers):
        """Create a default output structure when reference summaries aren't available"""
        default_output = {}
        
        for perspective, answers in perspective_answers.items():
            if answers:
                default_output[perspective] = "Summarize the " + perspective.lower() + " aspects from the answers."
        
        return default_output
    
    def _format_target_output(self, target_output):
        """Format the target output as a string"""
        output_text = "PERSPECTIVE SUMMARIES:\n\n"
        
        for perspective, summary in target_output.items():
            perspective_def = self.perspectives.get(perspective, {})
            start_phrase = perspective_def.get('start_phrase', f"{perspective}:")
            
            output_text += f"{perspective} SUMMARY: {start_phrase} {summary}\n\n"
        
        return output_text
            
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]