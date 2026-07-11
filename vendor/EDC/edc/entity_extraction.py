from typing import List
import os
from pathlib import Path
import edc.utils.llm_utils as llm_utils
import re
import copy


class EntityExtractor:
    # The class to handle the last stage: Schema Canonicalization
    def __init__(self, model=None, tokenizer=None, openai_model=None) -> None:
        # The canonicalizer uses an embedding model to first fetch candidates from the target schema, then uses a verifier schema to decide which one to canonicalize to or not
        # canonoicalize at all.

        assert openai_model is not None or (model is not None and tokenizer is not None)
        self.model = model
        self.tokenizer = tokenizer
        self.openai_model = openai_model

    def extract_entities(self, input_text_str: str, few_shot_examples_str: str, prompt_template_str: str):
        filled_prompt = prompt_template_str.format_map(
            {"few_shot_examples": few_shot_examples_str, "input_text": input_text_str}
        )
        messages = [{"role": "user", "content": filled_prompt}]

        if self.openai_model is None:
            # llm_utils.generate_completion_transformers([messages], self.model, self.tokenizer, device=self.device)
            completion = llm_utils.generate_completion_transformers(
                messages, self.model, self.tokenizer, answer_prepend="Entities: "
            )
        else:
            completion = llm_utils.openai_chat_completion(self.openai_model, None, messages)
        extracted_entities = llm_utils.parse_raw_entities(completion)
        return extracted_entities

    def merge_entities(
        self, input_text: str, entity_list_1: List[str], entity_list_2: List[str], prompt_template_str: str
    ):
        filled_prompt = prompt_template_str.format_map(
            {"input_text": input_text, "entity_list_1": entity_list_1, "entity_list_2": entity_list_2}
        )
        messages = [{"role": "user", "content": filled_prompt}]

        if self.openai_model is None:
            # llm_utils.generate_completion_transformers([messages], self.model, self.tokenizer, device=self.device)
            completion = llm_utils.generate_completion_transformers(
                messages, self.model, self.tokenizer, answer_prepend="Answer: "
            )
        else:
            completion = llm_utils.openai_chat_completion(self.openai_model, None, messages)
        extracted_entities = llm_utils.parse_raw_entities(completion)
        return extracted_entities
