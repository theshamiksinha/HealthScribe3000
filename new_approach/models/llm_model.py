from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import os
import yaml
import json

class PerspectiveLLM:
    def __init__(self, model_name_or_path, tokenizer=None, device=None):
        """
        Wrapper for LLM model for perspective-based summarization
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model and tokenizer
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path).to(self.device)
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(model_name_or_path)
        
        # Load configuration if available
        self.config = {}
        config_path = os.path.join(model_name_or_path, "config.yaml")
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        
    def generate(self, question, answers, max_length=512):
        """
        Generate perspective-based extraction and summaries for a question and its answers
        """
        # Filter out empty answers
        answers = [a for a in answers if a and a.strip() != '?']
        
        # Get perspective definitions
        perspectives = self.config.get('perspectives', {})
        
        # Create formatted prompt
        prompt = f"Question: {question}\n\n"
        prompt += "TASK: Extract relevant text spans for each perspective and generate perspective summaries.\n\n"
        
        # Add definitions for all perspectives
        prompt += "Perspectives:\n"
        for p_name, p_info in perspectives.items():
            prompt += f"- {p_name}: {p_info['definition']} (Tone: {p_info['tone']})\n"
        
        # Add answers
        prompt += "\nAnswers:\n"
        for i, answer in enumerate(answers):
            prompt += f"\nAnswer {i+1}: {answer}\n"
        
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
        
        # Parse output into sections
        result = self._parse_output(decoded_output)
        
        return result
    
    def _parse_output(self, output_text):
        """Parse the generated output into spans and summaries"""
        # Split output into sections
        result = {
            "extracted_spans": {},
            "summaries": {}
        }
        
        # Simple parsing by section headings
        sections = output_text.split("PERSPECTIVE SUMMARIES:")
        
        if len(sections) >= 2:
            spans_section = sections[0].replace("EXTRACTED SPANS:", "").strip()
            summaries_section = sections[1].strip()
            
            # Parse spans section
            current_perspective = None
            current_spans = []
            
            for line in spans_section.split("\n"):
                line = line.strip()
                if not line:
                    continue
                    
                if line.endswith(":") and not line.startswith("-"):
                    # Save previous perspective spans
                    if current_perspective and current_spans:
                        result["extracted_spans"][current_perspective] = current_spans
                    
                    # Start new perspective
                    current_perspective = line[:-1]  # Remove colon
                    current_spans = []
                elif line.startswith("- ") and current_perspective:
                    current_spans.append(line[2:])  # Remove "- " prefix
            
            # Save last perspective spans
            if current_perspective and current_spans:
                result["extracted_spans"][current_perspective] = current_spans
            
            # Parse summaries section
            for line in summaries_section.split("\n"):
                if "_SUMMARY:" in line:
                    parts = line.split("_SUMMARY:", 1)
                    perspective = parts[0].strip()
                    summary = parts[1].strip()
                    result["summaries"][perspective] = summary
        
        return result
    
    def save_prediction(self, output, filename):
        """Save prediction to file"""
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)