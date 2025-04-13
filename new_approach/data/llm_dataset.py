import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import random

class LLMDataset(Dataset):
    def __init__(self, data, tokenizer, config, mode="train"):
        """
        Dataset for training LLMs to extract and summarize perspective-specific content.
        
        Args:
            data: List of data instances
            tokenizer: HuggingFace tokenizer
            config: Configuration dictionary
            mode: "train", "val", or "test"
        """
        self.data = data
        self.tokenizer = tokenizer
        self.config = config
        self.mode = mode
        self.max_length = config['model']['llm']['max_length']
        # In this design, we assume the config's "perspectives" field is a dictionary 
        # with keys like "INFORMATION", "CAUSE", etc., mapping to their definitions, tones, etc.
        self.perspectives = config['perspectives']
        self.examples = self.preprocess()
        
    def preprocess(self):
        """Process raw data into model-ready examples."""
        examples = []
        max_pos_embeds = 1024  # Pegasus default
        vocab_size = self.tokenizer.vocab_size  # For clamping

        for idx, instance in enumerate(self.data):
            question = instance["question"]
            raw_text = instance["raw_text"]
            answers = instance["answers"]
            
            # Filter out empty answers or '?'
            answers = [a for a in answers if a and a.strip() != '?']
            
            # Get labelled spans and summaries
            labelled_spans = instance.get("labelled_answer_spans", {})
            labelled_summaries = instance.get("labelled_summaries", {})
            
            # Map each answer to its identified perspectives
            answer_perspectives = self._identify_answer_perspectives(raw_text, answers, labelled_spans)
            
            # Create the input prompt using question and answers with detected perspectives
            input_prompt = self._create_input_prompt(question, answer_perspectives)
            
            # Create target output using extracted spans and summaries
            target_output = self._create_target_output(labelled_spans, labelled_summaries)
            
            # Tokenize inputs with truncation and padding
            inputs = self.tokenizer(
                input_prompt,
                max_length=min(self.max_length, max_pos_embeds),
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )

            targets = self.tokenizer(
                target_output,
                max_length=min(self.max_length, max_pos_embeds),
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )

            # Clamp token IDs to ensure they don't exceed vocab size
            inputs["input_ids"] = torch.clamp(inputs["input_ids"], max=vocab_size - 1)
            targets["input_ids"] = torch.clamp(targets["input_ids"], max=vocab_size - 1)

            # Prepare labels; set pad tokens to -100 to ignore them in loss calculation
            labels = targets["input_ids"][0].clone()
            labels[labels == self.tokenizer.pad_token_id] = -100

            examples.append({
                "input_ids": inputs["input_ids"][0],
                "attention_mask": inputs["attention_mask"][0],
                "labels": labels,
            })

        return examples
        
    def _identify_answer_perspectives(self, raw_text, answers, labelled_spans):
        """Identify which perspectives are present in each answer."""
        answer_perspectives = {}
        
        for answer in answers:
            answer_start = raw_text.find(answer)
            if answer_start == -1:
                continue
            answer_end = answer_start + len(answer)
            present_perspectives = set()
            
            for perspective, spans in labelled_spans.items():
                for span in spans:
                    span_start, span_end = span["label_spans"]
                    if (answer_start <= span_start < answer_end or 
                        answer_start < span_end <= answer_end or
                        span_start <= answer_start < span_end):
                        present_perspectives.add(perspective)
            
            if present_perspectives:
                answer_perspectives[answer] = list(present_perspectives)
        
        return answer_perspectives
        
    def _create_input_prompt(self, question, answer_perspectives):
        """Create the input prompt for the model."""
        prompt = f"Question: {question}\n\n"
        prompt += "TASK: Extract relevant text spans for each perspective and generate perspective summaries.\n\n"
        
        prompt += "Perspectives:\n"
        for p_name, p_info in self.perspectives.items():
            prompt += f"- {p_name}: {p_info['definition']} (Tone: {p_info['tone']})\n"
        
        prompt += "\nAnswers and their perspectives:\n"
        for answer, perspectives in answer_perspectives.items():
            if not perspectives:
                continue
            perspective_list = ", ".join(perspectives)
            prompt += f"\nAnswer: {answer}\nPerspectives: {perspective_list}\n"
        
        return prompt
        
    def _create_target_output(self, labelled_spans, labelled_summaries):
        """Create the target output for the model."""
        # Part 1: Extracted spans organized by perspective
        output = "EXTRACTED SPANS:\n"
        for p_name, spans in labelled_spans.items():
            if spans:
                output += f"{p_name}:\n"
                for span in spans:
                    output += f"- {span['txt']}\n"
        
        # Part 2: Perspective summaries
        output += "\nPERSPECTIVE SUMMARIES:\n"
        for p_name in self.perspectives:
            summary_key = f"{p_name}_SUMMARY"
            if summary_key in labelled_summaries:
                p_info = self.perspectives.get(p_name, {})
                start_phrase = p_info.get('start_phrase', f"{p_name}:")
                output += f"{summary_key}: {start_phrase} {labelled_summaries[summary_key]}\n"
        
        return output
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        # Return the preprocessed example directly
        return self.examples[idx]
