# models/llm_model.py
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import os
import yaml
import json
from utils.prompt_templates import create_extraction_prompt

class PerspectiveLLM:
    def __init__(self, model_name_or_path, tokenizer=None, device=None):
        """
        Wrapper for LLM model for perspective-based summarization
        
        Args:
            model_name_or_path: Base model name or path to fine-tuned model
            tokenizer: Tokenizer (if None, will load based on model_name)
            device: Device to use for inference
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model and tokenizer
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path).to(self.device)
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(model_name_or_path)
        
        # Load configuration if available
        self.config = {}
        config_path = os.path.join(os.path.dirname(model_name_or_path), "config.yaml") \
            if not model_name_or_path.startswith("google/") else None
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        
    def generate(self, question, answers, perspectives=None, max_length=256):
        """
        Generate perspective-based summaries for a question and its answers
        
        Args:
            question: Question text
            answers: List of answer texts
            perspectives: List of perspectives to focus on (if None, use all)
            max_length: Maximum output length
            
        Returns:
            Dictionary of perspective-based summaries
        """
        # Prepare input with all answers
        all_perspectives = self.config.get('perspectives', {})
        if perspectives is None:
            perspectives = list(all_perspectives.keys())
        
        # Create formatted prompt
        prompt = create_extraction_prompt(
            question=question,
            answers=answers,
            perspectives={p: all_perspectives[p] for p in perspectives if p in all_perspectives}
        )
        
        # Tokenize input
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=1024
        ).to(self.device)
        
        # Generate output
        outputs = self.model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_length,
            num_beams=4,
            early_stopping=True
        )
        
        # Decode output
        decoded_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Parse output into perspective summaries
        summaries = self._parse_generated_output(decoded_output)
        
        return summaries
    
    def _parse_generated_output(self, output_text):
        """Parse the generated output into perspective summaries"""
        summaries = {}
        
        # Split by perspective markers
        lines = output_text.split('\n')
        current_perspective = None
        current_text = []
        
        for line in lines:
            if "SUMMARY:" in line:
                # Save previous perspective if any
                if current_perspective and current_text:
                    summaries[current_perspective] = " ".join(current_text).strip()
                    current_text = []
                
                # Extract new perspective
                perspective_part = line.split("SUMMARY:")[0].strip()
                current_perspective = perspective_part.strip()
                
                # Extract this line's text (after "SUMMARY:")
                text_part = line.split("SUMMARY:")[1].strip()
                if text_part:
                    current_text.append(text_part)
            
            elif current_perspective:
                current_text.append(line.strip())
        
        # Save last perspective
        if current_perspective and current_text:
            summaries[current_perspective] = " ".join(current_text).strip()
        
        return summaries