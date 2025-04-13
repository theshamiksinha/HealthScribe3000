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
        Dataset for training LLMs to summarize answers based on a specific perspective.
        
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
        examples = []
        max_pos_embeds = 456
        vocab_size = self.tokenizer.vocab_size

        for idx, instance in enumerate(self.data):
            question = instance["question"]
            raw_text = instance["raw_text"]
            answers = [a for a in instance["answers"] if a and a.strip() != '?']
            
            labelled_spans = instance.get("labelled_answer_spans", {})
            labelled_summaries = instance.get("labelled_summaries", {})

            # For each perspective, create a separate example
            for perspective, perspective_info in self.perspectives.items():
                relevant_spans = self._get_relevant_spans_for_perspective(perspective, labelled_spans, answers)
                
                if not relevant_spans:
                    continue  # Skip if no spans were found for this perspective

                input_prompt = self._create_input_prompt(question, perspective, perspective_info, relevant_spans)
                target_output = self._create_target_output(labelled_summaries, perspective)

                # Tokenizing input and target
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

                labels = targets["input_ids"][0].clone()
                labels[labels == self.tokenizer.pad_token_id] = -100

                examples.append({
                    "input_ids": inputs["input_ids"][0],
                    "attention_mask": inputs["attention_mask"][0],
                    "labels": labels,
                    "perspective": perspective  # ✅ This enables perspective-wise evaluation
                })


        return examples

    def _get_relevant_spans_for_perspective(self, perspective, labelled_spans, answers):
        """Retrieve the spans from answers that are relevant to a given perspective."""
        relevant_spans = []

        for answer in answers:
            if perspective in labelled_spans:
                for span in labelled_spans[perspective]:
                    if span["txt"] in answer:  # Match the span with the answer text
                        relevant_spans.append(span["txt"])

        return relevant_spans

    def _create_input_prompt(self, question, perspective, perspective_info, relevant_spans):
        prompt = f"Summarize the responses to the health question below.\n"
        prompt += f"Focus on highlighting insights from the {perspective} perspective.\n"
        prompt += f"Use a {perspective_info['tone']} tone. Be clear and concise.\n\n"
        prompt += f"Perspective Definition: {perspective_info['definition']}\n\n"
        prompt += f"Question: {question}\n\n"
        prompt += f"Answers:\n"
        for span in relevant_spans:
            prompt += f"- {span}\n"
        return prompt
 
    def _create_target_output(self, relevant_spans, perspective):
        """Create the target output for the model: summarize the spans."""
        # Here, you will likely want to concatenate the spans and summarize them.
        summary = " ".join(relevant_spans)  # For simplicity, we'll just join them.
        
        return f"{perspective}_SUMMARY: {summary}"
    
    def _create_target_output(self, labelled_summaries, persepective):
        # """Create the target output for the model."""
        # # Part 1: Extracted spans organized by perspective
        # output = "EXTRACTED SPANS:\n"
        # for p_name, spans in labelled_spans.items():
        #     if spans:
        #         output += f"{p_name}:\n"
        #         for span in spans:
        #             output += f"- {span['txt']}\n"
        
        # Part 2: Perspective summaries
        # output += "\nPERSPECTIVE SUMMARIES:\n"
        output = ""
        for p_name in self.perspectives:
            if persepective == p_name :
                summary_key = f"{p_name}_SUMMARY"
                if summary_key in labelled_summaries:
                    p_info = self.perspectives.get(p_name, {})
                    # start_phrase = p_info.get('start_phrase', f"{p_name}:")
                    start_phrase = ""
                    output += f"{summary_key}: {start_phrase} {labelled_summaries[summary_key]}\n"
        
        return output

    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        # Return the preprocessed example directly
        return self.examples[idx]
