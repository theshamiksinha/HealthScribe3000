import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import random

class LLMDataset(Dataset):
    def __init__(self, data, tokenizer, config, mode="train"):
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

        for instance in self.data:
            question = instance["question"]
            answers = instance.get("answers", [])
            labelled_spans = instance.get("labelled_answer_spans", {})
            labelled_summaries = instance.get("labelled_summaries", {})
            predicted_perspectives = instance.get("predicted_perspectives", [])

            # Decide which perspectives to use
            if self.mode in ["train", "val"]:
                perspectives_to_use = labelled_spans.keys()
            else:  # test mode
                perspectives_to_use = predicted_perspectives

            for perspective in perspectives_to_use:
                perspective_info = self.perspectives[perspective]

                if self.mode in ["train", "val"]:
                    relevant_spans = self._get_relevant_spans_for_perspective(perspective, labelled_spans, answers)
                    if not relevant_spans:
                        continue
                else:
                    # At test time, we don’t have labelled spans, so use full answer texts
                    relevant_spans = [ans for ans in answers if ans and ans.strip() != '?']
                    if not relevant_spans:
                        continue

                input_prompt = self._create_input_prompt(question, perspective, perspective_info, relevant_spans)

                inputs = self.tokenizer(
                    input_prompt,
                    max_length=min(self.max_length, max_pos_embeds),
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                )

                if self.mode in ["train", "val"]:
                    target_output = self._create_target_output(labelled_summaries, perspective)
                    targets = self.tokenizer(
                        target_output,
                        max_length=min(self.max_length, max_pos_embeds),
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt"
                    )
                    labels = targets["input_ids"][0].clone()
                    labels[labels == self.tokenizer.pad_token_id] = -100
                else:
                    target_output = self._create_target_output(labelled_summaries, perspective)
                    if target_output.strip():  # Only tokenize if non-empty
                        targets = self.tokenizer(
                            target_output,
                            max_length=min(self.max_length, max_pos_embeds),
                            padding="max_length",
                            truncation=True,
                            return_tensors="pt"
                        )
                        labels = targets["input_ids"][0].clone()
                        labels[labels == self.tokenizer.pad_token_id] = -100
                    else:
                        labels = None

                example = {
                    "input_ids": inputs["input_ids"][0],
                    "attention_mask": inputs["attention_mask"][0],
                    "perspective": perspective,
                }
                if labels is not None:
                    example["labels"] = labels

                examples.append(example)

        return examples

    def _get_relevant_spans_for_perspective(self, perspective, labelled_spans, answers):
        relevant_spans = []
        for answer in answers:
            if perspective in labelled_spans:
                for span in labelled_spans[perspective]:
                    if span["txt"] in answer:
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

    def _create_target_output(self, labelled_summaries, perspective):
        summary_key = f"{perspective}_SUMMARY"
        if summary_key in labelled_summaries:
            return f"{summary_key}: {labelled_summaries[summary_key]}"
        else:
            return ""

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]
