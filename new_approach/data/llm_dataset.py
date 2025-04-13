import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import random

class LLMDataset(Dataset):
    def __init__(self, data, tokenizer, config, mode="train"):
        """
        Dataset for training LLMs to extract and summarize perspective-specific content
        
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
        self.perspectives = config['perspectives']
        self.examples = self.preprocess()
        
    def preprocess(self):
        """Process raw data into model-ready examples"""
        examples = []
        
        for idx, instance in enumerate(self.data):
            question = instance["question"]
            raw_text = instance["raw_text"]
            answers = instance["answers"]
            
            # Filter out empty answers or '?'
            answers = [a for a in answers if a and a.strip() != '?']
            
            # Get labelled spans and summaries
            labelled_spans = instance.get("labelled_answer_spans", {})
            labelled_summaries = instance.get("labelled_summaries", {})
            
            # Map answers to their perspectives
            answer_perspectives = self._identify_answer_perspectives(raw_text, answers, labelled_spans)
            
            # Create input prompt
            input_prompt = self._create_input_prompt(question, answer_perspectives)
            
            # Create target output
            target_output = self._create_target_output(labelled_spans, labelled_summaries)
            
            # Tokenize input and target
            inputs = self.tokenizer(
                input_prompt,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            targets = self.tokenizer(
                target_output,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            # Create example
            example = {
                "instance_id": idx,
                "input_ids": inputs["input_ids"][0],
                "attention_mask": inputs["attention_mask"][0],
                "labels": targets["input_ids"][0],
                "question": question,
                "answers": answers,
                "perspectives": list(set(p for persp_list in answer_perspectives.values() for p in persp_list))
            }
            
            examples.append(example)
            
        return examples
        
    def _identify_answer_perspectives(self, raw_text, answers, labelled_spans):
        """Identify which perspectives are present in each answer"""
        answer_perspectives = {}
        
        # Extract perspective-related spans for each answer
        for answer in answers:
            # Find the position of this answer in the raw text
            answer_start = raw_text.find(answer)
            
            # Skip if answer not found in raw text
            if answer_start == -1:
                continue
                
            answer_end = answer_start + len(answer)
            present_perspectives = set()
            
            # Check which perspectives' spans overlap with this answer
            for perspective, spans in labelled_spans.items():
                for span in spans:
                    span_start, span_end = span["label_spans"]
                    
                    # Check if this span overlaps with the answer
                    if (answer_start <= span_start < answer_end or 
                        answer_start < span_end <= answer_end or
                        span_start <= answer_start < span_end):
                        present_perspectives.add(perspective)
            
            # Store perspectives for this answer
            if present_perspectives:
                answer_perspectives[answer] = list(present_perspectives)
        
        return answer_perspectives
        
    def _create_input_prompt(self, question, answer_perspectives):
        """Create the input prompt for the model"""
        prompt = f"Question: {question}\n\n"
        prompt += "TASK: Extract relevant text spans for each perspective and generate perspective summaries.\n\n"
        
        # Add definitions for all perspectives
        prompt += "Perspectives:\n"
        for p_name, p_info in self.perspectives.items():
            prompt += f"- {p_name}: {p_info['definition']} (Tone: {p_info['tone']})\n"
        
        # Add answers with their identified perspectives
        prompt += "\nAnswers with their perspectives:\n"
        for i, (answer, perspectives) in enumerate(answer_perspectives.items()):
            if perspectives:
                persp_str = ", ".join(perspectives)
                prompt += f"\nAnswer {i+1}: {answer}\n" 
                prompt += f"Perspectives: {persp_str}\n"
        
        return prompt
        
    def _create_target_output(self, labelled_spans, labelled_summaries):
        """Create the target output for the model"""
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
        return self.examples[idx]