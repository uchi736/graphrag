from typing import List
import os
import edc.utils.llm_utils as llm_utils
import numpy as np
import copy
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


class TypeCanonicalizer:
    # The class to canonicalize entity types, mirroring SchemaCanonicalizer.
    def __init__(
        self,
        target_type_dict: dict,
        embedder,
        verify_model=None,
        verify_tokenizer=None,
        verify_openai_model=None,
    ) -> None:
        # Uses an embedding model to retrieve candidate types, then an LLM verifier
        # to choose among them (or reject all).
        assert verify_openai_model is not None or (verify_model is not None and verify_tokenizer is not None)
        self.verifier_model = verify_model
        self.verifier_tokenizer = verify_tokenizer
        self.verifier_openai_model = verify_openai_model
        self.type_dict = target_type_dict

        self.embedder = embedder

        # Embed target types (using descriptions, same pattern as SC).
        self.type_embedding_dict = {}

        print("Embedding target types...")
        for type_name, type_description in tqdm(target_type_dict.items()):
            embedding = self.embedder.encode(type_description)
            self.type_embedding_dict[type_name] = embedding

    def retrieve_similar_types(self, query_type_or_description: str, top_k=5):
        target_type_list = list(self.type_embedding_dict.keys())
        target_type_embedding_list = list(self.type_embedding_dict.values())
        if "sts_query" in self.embedder.prompts:
            query_embedding = self.embedder.encode(query_type_or_description, prompt_name="sts_query")
        else:
            query_embedding = self.embedder.encode(query_type_or_description)

        # コサイン類似度（埋め込みを正規化）。閾値ゲートで使うため。
        q = np.asarray(query_embedding, dtype=float)
        M = np.asarray(target_type_embedding_list, dtype=float)
        q = q / (np.linalg.norm(q) + 1e-12)
        M = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)
        scores = M @ q

        highest_score_indices = np.argsort(-scores)

        return {
            target_type_list[idx]: self.type_dict[target_type_list[idx]]
            for idx in highest_score_indices[:top_k]
        }, [float(scores[idx]) for idx in highest_score_indices[:top_k]]

    def llm_verify(
        self,
        input_text_str: str,
        entity_name: str,
        raw_type: str,
        prompt_template_str: str,
        candidate_type_description_dict: dict,
    ):
        choice_letters_list = []
        choices = ""
        candidate_types = list(candidate_type_description_dict.keys())
        candidate_type_descriptions = list(candidate_type_description_dict.values())
        for idx, t in enumerate(candidate_types):
            choice_letter = chr(ord("@") + idx + 1)
            choice_letters_list.append(choice_letter)
            choices += f"{choice_letter}. '{t}': {candidate_type_descriptions[idx]}\n"
        choices += f"{chr(ord('@')+idx+2)}. None of the above.\n"

        verification_prompt = prompt_template_str.format_map(
            {
                "input_text": input_text_str,
                "entity": entity_name,
                "raw_type": raw_type,
                "choices": choices,
            }
        )

        messages = [{"role": "user", "content": verification_prompt}]
        if self.verifier_openai_model is None:
            verification_result = llm_utils.generate_completion_transformers(
                messages, self.verifier_model, self.verifier_tokenizer, answer_prepend="Answer: ", max_new_token=5
            )
        else:
            verification_result = llm_utils.openai_chat_completion(
                self.verifier_openai_model, None, messages, max_tokens=512
            )

        if verification_result and verification_result[0] in choice_letters_list:
            return candidate_types[choice_letters_list.index(verification_result[0])]
        return None

    def canonicalize(
        self,
        input_text_str: str,
        entity_name: str,
        raw_type: str,
        verify_prompt_template: str,
        enrich=False,
    ):
        if raw_type in self.type_dict:
            # Already canonical
            return raw_type, {}

        candidate_types = []
        candidate_scores = []
        canonicalized_type = None

        if len(self.type_dict) != 0:
            candidate_types, candidate_scores = self.retrieve_similar_types(raw_type)
            canonicalized_type = self.llm_verify(
                input_text_str,
                entity_name,
                raw_type,
                verify_prompt_template,
                candidate_types,
            )

        if canonicalized_type is None:
            # 類似度マージゲート（SchemaCanonicalizer と対称）: LLMが「該当なし」でも、
            # 最近傍の既存型とのコサイン類似度が閾値τ以上なら新規追加せず最近傍へ統合。
            merge_threshold = float(os.environ.get("CANON_MERGE_THRESHOLD", "0.9"))
            if candidate_types and candidate_scores and candidate_scores[0] >= merge_threshold:
                canonicalized_type = next(iter(candidate_types))
            elif enrich:
                # Add unknown type to the vocabulary (mirrors enrich_schema).
                definition = raw_type
                self.type_dict[raw_type] = definition
                if "sts_query" in self.embedder.prompts:
                    embedding = self.embedder.encode(definition, prompt_name="sts_query")
                else:
                    embedding = self.embedder.encode(definition)
                self.type_embedding_dict[raw_type] = embedding
                canonicalized_type = raw_type
        return canonicalized_type, dict(zip(candidate_types, candidate_scores))
