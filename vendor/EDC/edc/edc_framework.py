from edc.extract import Extractor
from edc.schema_definition import SchemaDefiner
from edc.schema_canonicalization import SchemaCanonicalizer
from edc.type_canonicalization import TypeCanonicalizer
from edc.entity_extraction import EntityExtractor
import edc.utils.llm_utils as llm_utils
from typing import List
from edc.schema_retriever import SchemaRetriever
from tqdm import tqdm
import os
import csv
import pathlib
from functools import partial
import copy
import logging
from importlib import reload
import random
import json

# Lazy imports for HuggingFace models (only needed for local models)
AutoModelForCausalLM = None
AutoTokenizer = None

reload(logging)
logger = logging.getLogger(__name__)


class EDC:
    def __init__(self, **edc_configuration) -> None:
        # OIE module settings
        self.oie_llm_name = edc_configuration["oie_llm"]
        self.oie_prompt_template_file_path = edc_configuration["oie_prompt_template_file_path"]
        self.oie_few_shot_example_file_path = edc_configuration["oie_few_shot_example_file_path"]

        # Schema Definition module settings
        self.sd_llm_name = edc_configuration["sd_llm"]
        self.sd_template_file_path = edc_configuration["sd_prompt_template_file_path"]
        self.sd_few_shot_example_file_path = edc_configuration["sd_few_shot_example_file_path"]

        # Schema Canonicalization module settings
        self.sc_llm_name = edc_configuration["sc_llm"]
        self.sc_embedder_name = edc_configuration["sc_embedder"]
        self.sc_template_file_path = edc_configuration["sc_prompt_template_file_path"]

        # Refinement settings
        self.sr_adapter_path = edc_configuration["sr_adapter_path"]

        self.sr_embedder_name = edc_configuration["sr_embedder"]
        self.oie_r_prompt_template_file_path = edc_configuration["oie_refine_prompt_template_file_path"]
        self.oie_r_few_shot_example_file_path = edc_configuration["oie_refine_few_shot_example_file_path"]

        self.ee_llm_name = edc_configuration["ee_llm"]
        self.ee_template_file_path = edc_configuration["ee_prompt_template_file_path"]
        self.ee_few_shot_example_file_path = edc_configuration["ee_few_shot_example_file_path"]

        self.em_template_file_path = edc_configuration["em_prompt_template_file_path"]

        self.initial_schema_path = edc_configuration["target_schema_path"]
        self.enrich_schema = edc_configuration["enrich_schema"]

        # Type Canonicalization module settings (symmetric to Schema Canonicalization)
        self.tc_template_file_path = edc_configuration.get(
            "tc_prompt_template_file_path", "./prompt_templates/tc_template.txt"
        )
        self.initial_types_path = edc_configuration.get("target_types_path", None)
        self.enrich_types = edc_configuration.get("enrich_types", False)
        self.oie_typed_template_file_path = edc_configuration.get(
            "oie_typed_prompt_template_file_path",
            "./prompt_templates/oie_typed_template.txt",
        )
        self.oie_typed_few_shot_example_file_path = edc_configuration.get(
            "oie_typed_few_shot_example_file_path",
            "./few_shot_examples/example/oie_typed_few_shot_examples.txt",
        )

        if self.initial_schema_path is not None:
            reader = csv.reader(open(self.initial_schema_path, "r", encoding="utf-8"))
            self.schema = {}
            for row in reader:
                relation, relation_definition = row
                self.schema[relation] = relation_definition
        else:
            self.schema = {}

        if self.initial_types_path is not None:
            reader = csv.reader(open(self.initial_types_path, "r", encoding="utf-8"))
            self.types = {}
            for row in reader:
                type_name, type_definition = row
                self.types[type_name] = type_definition
        else:
            self.types = {}

        # Load the needed models and tokenizers
        self.needed_model_set = set(
            [self.oie_llm_name, self.sd_llm_name, self.sc_llm_name, self.sc_embedder_name, self.ee_llm_name]
        )

        self.loaded_model_dict = {}

        logging.basicConfig(level=edc_configuration["loglevel"])

        logger.info(f"Model used: {self.needed_model_set}")

    def oie(
        self, input_text_list: List[str], previous_extracted_triplets_list: List[List[str]] = None, free_model=False
    ):
        if not llm_utils.is_model_openai(self.oie_llm_name):
            # Load the HF model for OIE
            oie_model, oie_tokenizer = self.load_model(self.oie_llm_name, "hf")
            # if self.oie_llm_name not in self.loaded_model_dict:
            #     logger.info(f"Loading model {self.oie_llm_name}.")
            #     oie_model, oie_tokenizer = (
            #         AutoModelForCausalLM.from_pretrained(self.oie_llm_name, device_map="auto"),
            #         AutoTokenizer.from_pretrained(self.oie_llm_name),
            #     )
            #     self.loaded_model_dict[self.oie_llm_name] = (oie_model, oie_tokenizer)
            # else:
            #     logger.info(f"Model {self.oie_llm_name} is already loaded, reusing it.")
            #     oie_model, oie_tokenizer = self.loaded_model_dict[self.oie_llm_name]
            extractor = Extractor(oie_model, oie_tokenizer)
        else:
            extractor = Extractor(openai_model=self.oie_llm_name)

        oie_triples_list = []
        entity_hint_list = None
        relation_hint_list = None

        if previous_extracted_triplets_list is not None:
            # Refined OIE
            logger.info("Running Refined OIE...")
            oie_refinement_prompt_template_str = open(self.oie_r_prompt_template_file_path, encoding="utf-8").read()
            oie_refinement_few_shot_examples_str = open(self.oie_r_few_shot_example_file_path, encoding="utf-8").read()

            logger.info("Putting together the refinement hint...")
            entity_hint_list, relation_hint_list = self.construct_refinement_hint(
                input_text_list, previous_extracted_triplets_list, free_model=free_model
            )

            assert len(previous_extracted_triplets_list) == len(input_text_list)
            for idx, input_text in enumerate(tqdm(input_text_list)):
                input_text = input_text_list[idx]
                entity_hint_str = entity_hint_list[idx]
                relation_hint_str = relation_hint_list[idx]
                refined_oie_triplets = extractor.extract(
                    input_text,
                    oie_refinement_few_shot_examples_str,
                    oie_refinement_prompt_template_str,
                    entity_hint_str,
                    relation_hint_str,
                )
                oie_triples_list.append(refined_oie_triplets)
        else:
            # Normal OIE
            entity_hint_list = ["" for _ in input_text_list]
            relation_hint_list = ["" for _ in input_text_list]
            logger.info("Running OIE...")

            # Typed mode: enabled when a target types schema is provided OR enrich_types is set
            # (mirrors how SC uses target_schema_path + enrich_schema)
            typed_mode = (self.initial_types_path is not None) or self.enrich_types
            entity_types_str = None
            if typed_mode:
                oie_few_shot_examples_str = open(
                    self.oie_typed_few_shot_example_file_path, encoding="utf-8"
                ).read()
                oie_few_shot_prompt_template_str = open(
                    self.oie_typed_template_file_path, encoding="utf-8"
                ).read()
                type_lines = [f"- {t}: {desc}" for t, desc in self.types.items()]
                entity_types_str = "\n".join(type_lines) if type_lines else "(no predefined types; propose type names freely)"
                logger.info(f"Typed OIE enabled with {len(type_lines)} predefined entity types")
            else:
                oie_few_shot_examples_str = open(self.oie_few_shot_example_file_path, encoding="utf-8").read()
                oie_few_shot_prompt_template_str = open(self.oie_prompt_template_file_path, encoding="utf-8").read()

            for input_text in tqdm(input_text_list):
                oie_triples = extractor.extract(
                    input_text,
                    oie_few_shot_examples_str,
                    oie_few_shot_prompt_template_str,
                    entity_types_str=entity_types_str,
                )
                oie_triples_list.append(oie_triples)
                logger.debug(f"{input_text}\n -> {oie_triples}\n")

        logger.info("OIE finished.")

        if free_model:
            logger.info(f"Freeing model {self.oie_llm_name} as it is no longer needed")
            llm_utils.free_model(oie_model, oie_tokenizer)
            del self.loaded_model_dict[self.oie_llm_name]

        return oie_triples_list, entity_hint_list, relation_hint_list

    def load_model(self, model_name, model_type):
        assert model_type in ["sts", "hf"]  # Either a sentence transformer or a huggingface LLM
        if model_name in self.loaded_model_dict:
            logger.info(f"Model {model_name} is already loaded, reusing it.")
        else:
            logger.info(f"Loading model {model_name}")
            if model_type == "hf":
                # Lazy import for HuggingFace models
                from transformers import AutoModelForCausalLM, AutoTokenizer
                model, tokenizer = (
                    AutoModelForCausalLM.from_pretrained(model_name, device_map="auto"),
                    AutoTokenizer.from_pretrained(model_name),
                )
                self.loaded_model_dict[model_name] = (model, tokenizer)
            elif model_type == "sts":
                # Use get_embedder to support Azure OpenAI embeddings
                model = llm_utils.get_embedder(model_name)
                self.loaded_model_dict[model_name] = model
        return self.loaded_model_dict[model_name]

    def schema_definition(self, input_text_list: List[str], oie_triplets_list: List[List[str]], free_model=False):
        assert len(input_text_list) == len(oie_triplets_list)

        if not llm_utils.is_model_openai(self.sd_llm_name):
            # Load the HF model for Schema Definition
            sd_model, sd_tokenizer = self.load_model(self.sd_llm_name, "hf")
            # if self.sd_llm_name not in self.loaded_model_dict:
            #     logger.info(f"Loading model {self.sd_llm_name}")
            #     sd_model, sd_tokenizer = (
            #         AutoModelForCausalLM.from_pretrained(self.sd_llm_name, device_map="auto"),
            #         AutoTokenizer.from_pretrained(self.sd_llm_name),
            #     )
            #     self.loaded_model_dict[self.sd_llm_name] = (sd_model, sd_tokenizer)
            #     logger.info(f"Loading model {self.sd_llm_name}.")
            # else:
            #     logger.info(f"Model {self.sd_llm_name} is already loaded, reusing it.")
            #     sd_model, sd_tokenizer = self.loaded_model_dict[self.sd_llm_name]
            schema_definer = SchemaDefiner(model=sd_model, tokenizer=sd_tokenizer)
        else:
            schema_definer = SchemaDefiner(openai_model=self.sd_llm_name)

        schema_definition_few_shot_prompt_template_str = open(self.sd_template_file_path, encoding="utf-8").read()
        schema_definition_few_shot_examples_str = open(self.sd_few_shot_example_file_path, encoding="utf-8").read()
        schema_definition_dict_list = []

        logger.info("Running Schema Definition...")
        for idx, oie_triplets in enumerate(tqdm(oie_triplets_list)):
            schema_definition_dict = schema_definer.define_schema(
                input_text_list[idx],
                oie_triplets,
                schema_definition_few_shot_examples_str,
                schema_definition_few_shot_prompt_template_str,
            )
            schema_definition_dict_list.append(schema_definition_dict)
            logger.debug(f"{input_text_list[idx]}, {oie_triplets}\n -> {schema_definition_dict}\n")

        logger.info("Schema Definition finished.")
        if free_model:
            logger.info(f"Freeing model {self.sd_llm_name} as it is no longer needed")
            llm_utils.free_model(sd_model, sd_tokenizer)
            del self.loaded_model_dict[self.sd_llm_name]
        return schema_definition_dict_list

    def schema_canonicalization(
        self,
        input_text_list: List[str],
        oie_triplets_list: List[List[str]],
        schema_definition_dict_list: List[dict],
        free_model=False,
    ):
        assert len(input_text_list) == len(oie_triplets_list) and len(input_text_list) == len(
            schema_definition_dict_list
        )
        logger.info("Running Schema Canonicalization...")

        sc_verify_prompt_template_str = open(self.sc_template_file_path, encoding="utf-8").read()

        # if self.sc_embedder_name not in self.loaded_model_dict:
        #     logger.info(f"Loading model {self.sc_embedder_name}.")
        #     sc_embedder = SentenceTransformer(self.sc_embedder_name, trust_remote_code=True)
        #     self.loaded_model_dict[self.sc_embedder_name] = sc_embedder

        # else:
        #     logger.info(f"Model {self.sc_embedder_name} is already loaded, reusing it.")
        #     sc_embedder = self.loaded_model_dict[self.sc_embedder_name]
        
        sc_embedder = self.load_model(self.sc_embedder_name, "sts")
        

        if not llm_utils.is_model_openai(self.sc_llm_name):
            sc_verify_model, sc_verify_tokenizer = self.load_model(self.sc_llm_name, "sts")
            # if self.sc_llm_name not in self.loaded_model_dict:
            #     logger.info(f"Loading model {self.sc_llm_name}")
            #     sc_verify_model, sc_verify_tokenizer = (
            #         AutoModelForCausalLM.from_pretrained(self.sc_llm_name, device_map="auto"),
            #         AutoTokenizer.from_pretrained(self.sc_llm_name),
            #     )
            #     self.loaded_model_dict[self.sc_llm_name] = (sc_verify_model, sc_verify_tokenizer)
            # else:
            #     logger.info(f"Model {self.sc_llm_name} is already loaded, reusing it.")
            #     sc_verify_model, sc_verify_tokenizer = self.loaded_model_dict[self.sc_llm_name]
            schema_canonicalizer = SchemaCanonicalizer(self.schema, sc_embedder, sc_verify_model, sc_verify_tokenizer)
        else:
            schema_canonicalizer = SchemaCanonicalizer(self.schema, sc_embedder, verify_openai_model=self.sc_llm_name)

        canonicalized_triplets_list = []
        canon_candidate_dict_per_entry_list = []

        for idx, input_text in enumerate(tqdm(input_text_list)):
            oie_triplets = oie_triplets_list[idx]
            canonicalized_triplets = []
            sd_dict = schema_definition_dict_list[idx]
            canon_candidate_dict_list = []
            for oie_triplet in oie_triplets:
                canonicalized_triplet, canon_candidate_dict = schema_canonicalizer.canonicalize(
                    input_text, oie_triplet, sd_dict, sc_verify_prompt_template_str, self.enrich_schema
                )
                canonicalized_triplets.append(canonicalized_triplet)
                canon_candidate_dict_list.append(canon_candidate_dict)

            canonicalized_triplets_list.append(canonicalized_triplets)
            canon_candidate_dict_per_entry_list.append(canon_candidate_dict_list)

            logger.debug(f"{input_text}\n, {oie_triplets} ->\n {canonicalized_triplets}")
            logger.debug(f"Retrieved candidate relations {canon_candidate_dict_list}")
        logger.info("Schema Canonicalization finished.")

        if free_model:
            logger.info(f"Freeing model {self.sc_embedder_name, self.sc_llm_name} as it is no longer needed")
            llm_utils.free_model(sc_embedder)
            llm_utils.free_model(sc_verify_model, sc_verify_tokenizer)
            del self.loaded_model_dict[self.sc_llm_name]

        return canonicalized_triplets_list, canon_candidate_dict_per_entry_list

    def type_canonicalization(
        self,
        input_text_list: List[str],
        canonicalized_triplets_list: List[List[List[str]]],
        free_model=False,
    ):
        """Normalize entity types in 5-tuple triplets against self.types.

        Mirror of schema_canonicalization(): takes canonicalized triplets and
        routes each (entity, raw_type) pair through a TypeCanonicalizer.
        3-tuple triplets (no types) are passed through unchanged.
        """
        logger.info("Running Type Canonicalization...")

        tc_verify_prompt_template_str = open(self.tc_template_file_path, encoding="utf-8").read()

        tc_embedder = self.load_model(self.sc_embedder_name, "sts")

        if not llm_utils.is_model_openai(self.sc_llm_name):
            tc_verify_model, tc_verify_tokenizer = self.load_model(self.sc_llm_name, "sts")
            type_canonicalizer = TypeCanonicalizer(self.types, tc_embedder, tc_verify_model, tc_verify_tokenizer)
        else:
            type_canonicalizer = TypeCanonicalizer(self.types, tc_embedder, verify_openai_model=self.sc_llm_name)

        typed_triplets_list = []
        type_candidate_dict_per_entry_list = []

        # Cache (entity, raw_type) → canonical type to avoid redundant LLM calls
        type_cache: dict = {}

        for idx, input_text in enumerate(tqdm(canonicalized_triplets_list)):
            chunk = canonicalized_triplets_list[idx]
            typed_chunk = []
            type_candidates_chunk = []
            for triplet in chunk:
                if triplet is None or len(triplet) != 5:
                    typed_chunk.append(triplet)
                    type_candidates_chunk.append({})
                    continue
                subj, subj_type_raw, rel, obj, obj_type_raw = triplet

                subj_key = (subj, subj_type_raw)
                if subj_key in type_cache:
                    subj_type = type_cache[subj_key]
                    subj_candidates = {}
                else:
                    subj_type, subj_candidates = type_canonicalizer.canonicalize(
                        input_text_list[idx], subj, subj_type_raw,
                        tc_verify_prompt_template_str, self.enrich_types,
                    )
                    if subj_type is None:
                        subj_type = "Other"
                    type_cache[subj_key] = subj_type

                obj_key = (obj, obj_type_raw)
                if obj_key in type_cache:
                    obj_type = type_cache[obj_key]
                    obj_candidates = {}
                else:
                    obj_type, obj_candidates = type_canonicalizer.canonicalize(
                        input_text_list[idx], obj, obj_type_raw,
                        tc_verify_prompt_template_str, self.enrich_types,
                    )
                    if obj_type is None:
                        obj_type = "Other"
                    type_cache[obj_key] = obj_type

                typed_chunk.append([subj, subj_type, rel, obj, obj_type])
                type_candidates_chunk.append({"subject": subj_candidates, "object": obj_candidates})

            typed_triplets_list.append(typed_chunk)
            type_candidate_dict_per_entry_list.append(type_candidates_chunk)

        logger.info("Type Canonicalization finished.")

        if free_model:
            logger.info(f"Freeing TC models")
            llm_utils.free_model(tc_embedder)

        return typed_triplets_list, type_candidate_dict_per_entry_list

    def construct_refinement_hint(
        self,
        input_text_list: List[str],
        extracted_triplets_list: List[List[List[str]]],
        include_relation_example="self",
        relation_top_k=10,
        free_model=False,
    ):
        entity_extraction_few_shot_examples_str = open(self.ee_few_shot_example_file_path, encoding="utf-8").read()
        entity_extraction_prompt_template_str = open(self.ee_template_file_path, encoding="utf-8").read()

        entity_merging_prompt_template_str = open(self.em_template_file_path, encoding="utf-8").read()

        entity_hint_list = []
        relation_hint_list = []

        # Initialize entity extractor
        if not llm_utils.is_model_openai(self.ee_llm_name):
            # Load the HF model for Schema Definition
            ee_model, ee_tokenizer = self.load_model(self.ee_llm_name, "hf")
            # if self.ee_llm_name not in self.loaded_model_dict:
            #     logger.info(f"Loading model {self.ee_llm_name}")
            #     ee_model, ee_tokenizer = (
            #         AutoModelForCausalLM.from_pretrained(self.ee_llm_name, device_map="auto"),
            #         AutoTokenizer.from_pretrained(self.ee_llm_name),
            #     )
            #     self.loaded_model_dict[self.ee_llm_name] = (ee_model, ee_tokenizer)
            # else:
            #     logger.info(f"Model {self.ee_llm_name} is already loaded, reusing it.")
            #     ee_model, ee_tokenizer = self.loaded_model_dict[self.ee_llm_name]
            entity_extractor = EntityExtractor(model=ee_model, tokenizer=ee_tokenizer)
        else:
            entity_extractor = EntityExtractor(openai_model=self.sd_llm_name)

        # Initialize schema retriever
        # if self.sr_embedder_name not in self.loaded_model_dict:
        #     logger.info(f"Loading model {self.sr_embedder_name}.")
        #     sr_embedding_model = SentenceTransformer(self.sr_embedder_name)
        #     self.loaded_model_dict[self.sr_embedder_name] = sr_embedding_model
        # else:
        #     sr_embedding_model = self.loaded_model_dict[self.sr_embedder_name]
        #     logger.info(f"Model {self.sr_embedder_name} is already loaded, reusing it.")
        sr_embedding_model = self.load_model(self.sr_embedder_name, "sts")

        schema_retriever = SchemaRetriever(
            self.schema,
            sr_embedding_model,
            None,
            finetuned_e5mistral=False,
        )

        relation_example_dict = {}
        if include_relation_example == "self":
            # Include an example of where this relation can be extracted
            for idx in range(len(input_text_list)):
                input_text_str = input_text_list[idx]
                extracted_triplets = extracted_triplets_list[idx]
                for triplet in extracted_triplets:
                    relation = triplet[1]
                    if relation not in relation_example_dict:
                        relation_example_dict[relation] = [{"text": input_text_str, "triplet": triplet}]
                    else:
                        relation_example_dict[relation].append({"text": input_text_str, "triplet": triplet})
        else:
            # Todo: allow to pass gold examples of relations
            pass

        for idx in tqdm(range(len(input_text_list))):
            input_text_str = input_text_list[idx]
            extracted_triplets = extracted_triplets_list[idx]

            previous_relations = set()
            previous_entities = set()

            for triplet in extracted_triplets:
                previous_entities.add(triplet[0])
                previous_entities.add(triplet[2])
                previous_relations.add(triplet[1])

            previous_entities = list(previous_entities)
            previous_relations = list(previous_relations)

            # Obtain candidate entities
            extracted_entities = entity_extractor.extract_entities(
                input_text_str, entity_extraction_few_shot_examples_str, entity_extraction_prompt_template_str
            )
            merged_entities = entity_extractor.merge_entities(
                input_text_str, previous_entities, extracted_entities, entity_merging_prompt_template_str
            )
            entity_hint_list.append(str(merged_entities))

            # Obtain candidate relations
            hint_relations = previous_relations

            retrieved_relations = schema_retriever.retrieve_relevant_relations(input_text_str)

            counter = 0

            for relation in retrieved_relations:
                if counter >= relation_top_k:
                    break
                else:
                    if relation not in hint_relations:
                        hint_relations.append(relation)

            candidate_relation_str = ""
            for relation_idx, relation in enumerate(hint_relations):
                if relation not in self.schema:
                    continue

                relation_definition = self.schema[relation]

                candidate_relation_str += f"{relation_idx+1}. {relation}: {relation_definition}\n"
                if include_relation_example == "self":
                    if relation not in relation_example_dict:
                        # candidate_relation_str += "Example: None.\n"
                        pass
                    else:
                        selected_example = None
                        if len(relation_example_dict[relation]) != 0:
                            selected_example = random.choice(relation_example_dict[relation])
                        # for example in relation_example_dict[relation]:
                        #     if example["text"] != input_text_str:
                        #         selected_example = example
                        #         break
                        if selected_example is not None:
                            candidate_relation_str += f"""For example, {selected_example['triplet']} can be extracted from "{selected_example['text']}"\n"""
                        else:
                            # candidate_relation_str += "Example: None.\n"
                            pass
            relation_hint_list.append(candidate_relation_str)

        if free_model:
            logger.info(f"Freeing model {self.sr_embedder_name, self.ee_llm_name} as it is no longer needed")
            llm_utils.free_model(sr_embedding_model)
            llm_utils.free_model(ee_model, ee_tokenizer)
            del self.loaded_model_dict[self.sr_embedder_name]
            del self.loaded_model_dict[self.ee_llm_name]
        return entity_hint_list, relation_hint_list

    def extract_kg(self, input_text_list: List[str], output_dir: str = None, refinement_iterations=0):
        if output_dir is not None:
            if os.path.exists(output_dir):
                import shutil
                logger.warning(f"Output directory {output_dir} already exists. Overwriting.")
                shutil.rmtree(output_dir)
            for iteration in range(refinement_iterations + 1):
                pathlib.Path(f"{output_dir}/iter{iteration}").mkdir(parents=True, exist_ok=True)


        # EDC run
        logger.info("EDC starts running...")

        required_model_dict = {
            "oie": self.oie_llm_name,
            "sd": self.sd_llm_name,
            "sc_embed": self.sc_embedder_name,
            "sc_verify": self.sc_llm_name,
            "ee": self.ee_llm_name,
            "sr": self.sr_embedder_name,
        }

        triplets_from_last_iteration = None
        for iteration in range(refinement_iterations + 1):
            logger.info(f"Iteration {iteration}:")

            iteration_result_dir = f"{output_dir}/iter{iteration}"

            required_model_dict_current_iteration = copy.deepcopy(required_model_dict)

            del required_model_dict_current_iteration["oie"]
            oie_triplets_list, entity_hint_list, relation_hint_list = self.oie(
                input_text_list,
                free_model=self.oie_llm_name not in required_model_dict_current_iteration.values()
                and iteration == refinement_iterations,
                previous_extracted_triplets_list=triplets_from_last_iteration,
            )

            # Typed mode: strip 5-tuples to 3-tuples before SD/SC (which assume 3-tuples).
            # Keep the original 5-tuples in a sidecar so types can be re-attached after SC.
            typed_mode = (self.initial_types_path is not None) or self.enrich_types
            typed_oie_list = None
            if typed_mode:
                typed_oie_list = oie_triplets_list
                oie_triplets_list = [
                    [
                        [t[0], t[2], t[3]] if len(t) == 5 else list(t)
                        for t in chunk
                    ]
                    for chunk in typed_oie_list
                ]

            del required_model_dict_current_iteration["sd"]
            sd_dict_list = self.schema_definition(
                input_text_list,
                oie_triplets_list,
                free_model=self.sd_llm_name not in required_model_dict_current_iteration.values()
                and iteration == refinement_iterations,
            )

            del required_model_dict_current_iteration["sc_embed"]
            del required_model_dict_current_iteration["sc_verify"]
            canon_triplets_list, canon_candidate_dict_list = self.schema_canonicalization(
                input_text_list,
                oie_triplets_list,
                sd_dict_list,
                free_model=self.sc_llm_name not in required_model_dict_current_iteration.values()
                and iteration == refinement_iterations,
            )

            # Typed mode: re-attach types (positional align with pre-SC triplets),
            # then run Type Canonicalization (symmetric to Schema Canonicalization).
            if typed_mode:
                restored_canon = []
                for chunk_idx, canon_chunk in enumerate(canon_triplets_list):
                    src_chunk = typed_oie_list[chunk_idx]
                    restored = []
                    for i, canon_t in enumerate(canon_chunk):
                        if canon_t is None:
                            restored.append(None)
                            continue
                        if i < len(src_chunk) and len(src_chunk[i]) == 5:
                            subj_type = src_chunk[i][1]
                            obj_type = src_chunk[i][4]
                            restored.append([canon_t[0], subj_type, canon_t[1], canon_t[2], obj_type])
                        else:
                            restored.append(canon_t)
                    restored_canon.append(restored)

                canon_triplets_list, type_candidate_dict_list = self.type_canonicalization(
                    input_text_list,
                    restored_canon,
                )

            non_null_triplets_list = [
                [triple for triple in triplets if triple is not None] for triplets in canon_triplets_list
            ]
            # for triplets in canon_triplets_list:
            #     non_null_triplets = []
            #     for triple in triplets:
            #         if triple is not None:
            #             non_n
            triplets_from_last_iteration = non_null_triplets_list

            # Write results
            assert len(oie_triplets_list) == len(sd_dict_list) and len(sd_dict_list) == len(canon_triplets_list)

            json_results_list = []
            for idx in range(len(oie_triplets_list)):
                result_json = {
                    "index": idx,
                    "input_text": input_text_list[idx],
                    "entity_hint": entity_hint_list[idx],
                    "relation_hint": relation_hint_list[idx],
                    "oie": oie_triplets_list[idx],
                    "schema_definition": sd_dict_list[idx],
                    "canonicalization_candidates": str(canon_candidate_dict_list[idx]),
                    "schema_canonicalizaiton": canon_triplets_list[idx],
                }
                json_results_list.append(result_json)
            result_at_each_stage_file = open(f"{iteration_result_dir}/result_at_each_stage.json", "w", encoding="utf-8")
            json.dump(json_results_list, result_at_each_stage_file, indent=4)

            final_result_file = open(f"{iteration_result_dir}/canon_kg.txt", "w", encoding="utf-8")
            for idx, canon_triplets in enumerate(non_null_triplets_list):
                final_result_file.write(str(canon_triplets))
                if idx != len(canon_triplets_list) - 1:
                    final_result_file.write("\n")
                final_result_file.flush()

        return canon_triplets_list

    # ------------------------------------------------------------------
    # HITL (Human-in-the-loop) 用の2段階API。
    # extract_kg() を Phase A / Phase B に分割し、間に人手のスキーマ・
    # キュレーションを挟めるようにする。各ステージ実装(oie/schema_definition/
    # schema_canonicalization/type_canonicalization)は extract_kg と共用。
    # ------------------------------------------------------------------
    def extract_and_define(self, input_text_list: List[str]):
        """Phase A: OIE + Schema Definition のみ実行し、発見スキーマを返す。

        Returns:
            dict: {
              "input_text_list", "oie_triplets_list", "sd_dict_list",
              "typed_oie_list" (typed時の5タプル, それ以外None), "typed_mode",
              "discovered_schema": [{"relation","definition","count"}, ...] (count降順),
              "discovered_types": [{"type","definition","count"}, ...] (typed時のみ。それ以外None)
            }
        """
        oie_triplets_list, _entity_hint_list, _relation_hint_list = self.oie(
            input_text_list, previous_extracted_triplets_list=None, free_model=False
        )

        # Typed mode: 5タプルを3タプルへ縮約し、元の5タプルをサイドカー保持
        # (extract_kg と同一ロジック)
        typed_mode = (self.initial_types_path is not None) or self.enrich_types
        typed_oie_list = None
        if typed_mode:
            typed_oie_list = oie_triplets_list
            oie_triplets_list = [
                [[t[0], t[2], t[3]] if len(t) == 5 else list(t) for t in chunk]
                for chunk in typed_oie_list
            ]

        sd_dict_list = self.schema_definition(input_text_list, oie_triplets_list, free_model=False)

        # 発見スキーマを構築: 定義は最長のものを採用、count は OIE 出現回数
        relation_def = {}
        for sd_dict in sd_dict_list:
            for rel, definition in sd_dict.items():
                definition = definition or ""
                if rel not in relation_def or len(definition) > len(relation_def[rel]):
                    relation_def[rel] = definition
        relation_count = {}
        for chunk in oie_triplets_list:
            for t in chunk:
                if len(t) >= 2:
                    rel = t[1]
                    relation_count[rel] = relation_count.get(rel, 0) + 1
        all_relations = set(relation_def) | set(relation_count)
        discovered_schema = [
            {
                "relation": rel,
                "definition": relation_def.get(rel, ""),
                "count": relation_count.get(rel, 0),
            }
            for rel in all_relations
        ]
        discovered_schema.sort(key=lambda x: -x["count"])

        # 発見された型を構築 (typed時のみ。relation スキーマと対称)。
        # 初期型スキーマ self.types の定義をシードに、typed OIE が出した raw 型を
        # 追加し、出現回数を集計する。OIE は型に定義を付けないので未シード型の
        # definition は空（人手キュレーションで補える）。
        discovered_types = None
        if typed_mode and typed_oie_list is not None:
            type_def = dict(self.types)  # 初期型スキーマの定義をシード
            type_count = {}
            for chunk in typed_oie_list:
                for t in chunk:
                    if len(t) == 5:
                        for tp in (t[1], t[4]):
                            type_count[tp] = type_count.get(tp, 0) + 1
                            type_def.setdefault(tp, "")
            all_types = set(type_def) | set(type_count)
            discovered_types = [
                {"type": tp, "definition": type_def.get(tp, ""), "count": type_count.get(tp, 0)}
                for tp in all_types
            ]
            discovered_types.sort(key=lambda x: -x["count"])

        return {
            "input_text_list": input_text_list,
            "oie_triplets_list": oie_triplets_list,
            "sd_dict_list": sd_dict_list,
            "typed_oie_list": typed_oie_list,
            "typed_mode": typed_mode,
            "discovered_schema": discovered_schema,
            "discovered_types": discovered_types,
        }

    def canonicalize_with_schema(
        self,
        input_text_list: List[str],
        oie_triplets_list,
        sd_dict_list,
        curated_schema: dict,
        typed_oie_list=None,
        enrich: bool = False,
        curated_types: dict = None,
        enrich_types: bool = False,
    ):
        """Phase B: 人手で確定したスキーマで Schema (+ Type) Canonicalization を実行。

        curated_schema (relation->definition) を self.schema に設定して再正規化する。
        SchemaCanonicalizer は呼び出し毎に self.schema を再embeddingするため、
        確定スキーマに対してマッピングされる。
        typed時 (typed_oie_list が渡された場合) は型の再付与 + Type Canonicalization も行う。
        curated_types を渡すと self.types に設定し、enrich_types=True なら確定型スキーマに
        無い型は新規追加（「初期を与えて漏れは追加」）。
        """
        self.schema = dict(curated_schema)
        self.enrich_schema = enrich

        canon_triplets_list, _canon_candidate_dict_list = self.schema_canonicalization(
            input_text_list, oie_triplets_list, sd_dict_list, free_model=False
        )

        # Typed mode: 型を再付与して Type Canonicalization (extract_kg と同一ロジック)。
        # typed_oie_list の有無で typed か判定（最も確実なシグナル）。
        if typed_oie_list is not None:
            if curated_types is not None:
                self.types = dict(curated_types)
            self.enrich_types = enrich_types
            restored_canon = []
            for chunk_idx, canon_chunk in enumerate(canon_triplets_list):
                src_chunk = typed_oie_list[chunk_idx]
                restored = []
                for i, canon_t in enumerate(canon_chunk):
                    if canon_t is None:
                        restored.append(None)
                        continue
                    if i < len(src_chunk) and len(src_chunk[i]) == 5:
                        subj_type = src_chunk[i][1]
                        obj_type = src_chunk[i][4]
                        restored.append([canon_t[0], subj_type, canon_t[1], canon_t[2], obj_type])
                    else:
                        restored.append(canon_t)
                restored_canon.append(restored)

            canon_triplets_list, _type_candidate_dict_list = self.type_canonicalization(
                input_text_list, restored_canon
            )

        return canon_triplets_list