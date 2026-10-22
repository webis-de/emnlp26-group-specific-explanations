import gc
import time
from typing import Dict, List, Optional, Union
import pandas as pd
import spacy
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BatchEncoding, StoppingCriteria, StoppingCriteriaList
import torch
import logging
import numpy as np
import re

from config_new_prompts import (
    ACTIVATION_SPACE_LAMBDAS,
    NUM_IMPORTANT_ATTRIBUTES_KNOWLEDGE,
    NUM_IMPORTANT_ATTRIBUTES_STYLE,
    STYLE_EMBEDDING_VECTOR,
    USE_IMPORTANT_ATTRIBUTES_FOR_ACTIVATION_STEERING,
    ACTIVATION_STEERING_INSERT_LAYER_INDECES,
    KONEN_ACTIVATION_SPACE_LAMBDAS
)
from helper import log_time
from scripts.steering.activation_steering.steering_layer import SteeringLayer
from scripts.steering.prompt_steering import (
    _get_most_important_attributes,
    convert_attribute_sentence,
    get_system_prompt_attribute,
    get_system_prompt_group,
    get_system_prompt_group_attribute
    
)
from scripts.evaluate import _evaluate_activation_steering_joint_score, plot_activation_steering_performance
from utils.database import Database

import deepspeed

logger = logging.getLogger(__name__)

def evaluate(steering_type, steering_model, questions, num_important_attributes):
    from scripts.steering.steering import (
        _create_style_vectors,
        _create_style_embeddings,
        _create_normal_embeddings

    )
    print("starting steering evaluation")
    database = Database()
    steered_data = database.get_steering_generation_of_llm(llm_name=steering_model)
    steered_data = steered_data[
        steered_data["question_id"].isin(questions["id"].values.tolist())
    ]

    answers = database.get_answers(
        ["sfam_train", "sfam_val", "sfam_test", "lisa_train", "lisa_val", "lisa_test"]
    )
    answers = answers.sort_values("data_group")
    
    # Embeddings based on LISA
    target_group_order_lisa = []
    group_embedding_series = answers.groupby("data_group")["style_embedding"].apply(list)
    group_mean_embeddings_lisa = []
    other_groups_mean_embeddings_lisa = []
    for group, group_embeddings in group_embedding_series.items():
        target_group_order_lisa.append(group)
        group_mean_embeddings_lisa.append(np.mean(group_embeddings, axis=0))
        all_other_groups_embedding_lisa = answers[answers["data_group"] != group]["style_embedding"].values.tolist()
        other_groups_mean_embeddings_lisa.append(np.mean(all_other_groups_embedding_lisa, axis=0))
    group_mean_embeddings_lisa = np.array(group_mean_embeddings_lisa)
    other_groups_mean_embeddings_lisa = None
    
    # Embeddings based on normal encoder 
    target_group_order_normal = []
    group_embedding_series = answers.groupby("data_group")["normal_embedding"].apply(list)
    group_mean_embeddings_normal = []
    other_groups_mean_embeddings_normal = []
    for group, group_embeddings in group_embedding_series.items():
        target_group_order_normal.append(group)
        group_mean_embeddings_normal.append(np.mean(group_embeddings, axis=0) / np.linalg.norm(np.mean(group_embeddings, axis=0)))
        all_other_groups_embedding_normal = answers[answers["data_group"] != group]["normal_embedding"].values.tolist()
        other_groups_mean_embeddings_normal.append(np.mean(all_other_groups_embedding_normal, axis=0) / np.linalg.norm(np.mean(all_other_groups_embedding_normal, axis=0)))
    group_mean_embeddings_normal = np.array(group_mean_embeddings_normal)
    other_groups_mean_embeddings_normal = None

    # Get steered data to create style vectors
    steered_data = steered_data.sort_values("data_group")
    base_steering = steered_data[steered_data["steering_type"] == "base"]
    steered_data_activation = steered_data[steered_data["steering_type"] == steering_type]

    _create_style_vectors(steering_model, steered_data=base_steering)
    _create_style_vectors(steering_model, steered_data=steered_data_activation)
    
    # Get steered data including style vectors to create style embeddings and normal embeddings
    steered_data = database.get_steering_generation_of_llm(llm_name=steering_model)
    steered_data = steered_data[
        steered_data["question_id"].isin(questions["id"].values.tolist())
    ]
    steered_data = steered_data.sort_values("data_group")
    base_steering = steered_data[steered_data["steering_type"] == "base"]
    steered_data_activation = steered_data[steered_data["steering_type"] == steering_type]
    
    _create_style_embeddings(steering_model, steered_data=base_steering)
    _create_normal_embeddings(steering_model, steered_data=base_steering)    
    _create_style_embeddings(steering_model, steered_data=steered_data_activation)
    _create_normal_embeddings(steering_model, steered_data=steered_data_activation)   
    
    #return None, None
    # Get steered data including all embeddings to get best performing hyperparameters (best_activation_layer+best_activation_lambda)
    steered_data = database.get_steering_generation_of_llm(llm_name=steering_model)
    steered_data = steered_data[
        steered_data["question_id"].isin(questions["id"].values.tolist())
    ]
    steered_data = steered_data.sort_values("data_group")
    
    base_steering = steered_data[steered_data["steering_type"] == "base"]
    
    steered_data_activation = steered_data[steered_data["steering_type"] == steering_type]
    steered_data_activation = steered_data_activation.sort_values("data_group")
    
    _, _, best_activation_layer, _, best_activation_lambda, _, target_group_order = _evaluate_activation_steering_joint_score(
        steering_type=steering_type,
        steering_model=steering_model,
        steered_data_df=steered_data_activation,
        base_steering=base_steering,
        group_mean_embeddings_lisa=group_mean_embeddings_lisa,
        group_mean_embeddings_normal=group_mean_embeddings_normal,
        other_groups_mean_embeddings_normal=other_groups_mean_embeddings_normal
    )
    
    # Check group order
    assert target_group_order_lisa == target_group_order
    assert target_group_order_normal == target_group_order
    
    print(
        f"Best activation layer: {best_activation_layer}, best activation lambda: {best_activation_lambda}",
    )
    return list(map(int, best_activation_layer.split(','))), best_activation_lambda

def group_steering(steering_model, num_important_attributes):
    database = Database()
    
    steering_type = f"activation_group_{num_important_attributes}"
    
    val_questions = database.get_steering_questions(split="val")
    #val_questions = val_questions[:10]
        
    run_activation_steering(steering_type, steering_model, val_questions, num_important_attributes)

    best_activation_layer, best_activation_lambda = evaluate(steering_type, steering_model, val_questions, num_important_attributes)
    
    test_questions = database.get_steering_questions(split="test")
    #test_questions = pd.concat([database.get_steering_questions(split="eli5_test")[:20], database.get_steering_questions(split="rooein_test")[:20]], ignore_index=True)
    
    run_activation_steering(steering_type, steering_model, test_questions, num_important_attributes, best_activation_layer=best_activation_layer, best_activation_lambda=best_activation_lambda)
    
    return steering_type, best_activation_layer, best_activation_lambda

def group_attribute_steering(steering_model, num_important_attributes):
    database = Database()
    
    steering_type = f"activation_group_attribute_{num_important_attributes}"
    
    val_questions = database.get_steering_questions(split="val")
    #val_questions = val_questions[:10]
    
    run_activation_steering(steering_type, steering_model, val_questions, num_important_attributes)

    best_activation_layer, best_activation_lambda = evaluate(steering_type, steering_model, val_questions, num_important_attributes)
    
    test_questions = database.get_steering_questions(split="test")
    #test_questions = pd.concat([database.get_steering_questions(split="eli5_test")[:20], database.get_steering_questions(split="rooein_test")[:20]], ignore_index=True)
    
    run_activation_steering(steering_type, steering_model, test_questions, num_important_attributes, best_activation_layer=best_activation_layer, best_activation_lambda=best_activation_lambda)
    
    return steering_type, best_activation_layer, best_activation_lambda


def group_steering_test_questions(steering_model, num_important_attributes):
    database = Database()
    
    steering_type = f"activation_group_{num_important_attributes}"
    
    test_questions = database.get_steering_questions(split="test")
    steered_data = database.get_steering_generation_of_llm(llm_name=steering_model)

    best_activation_layer, best_activation_lambda = None, None
    
    if database.file_name == "llama_data_qwen_filtered_sent.sqlite":
        best_activation_layer = [17, 18, 19]
        best_activation_lambda = 0.75
    else:
        for steering_type_index, (_, group) in enumerate(
            steered_data.groupby("steering_type")
        ):
            if group["steering_type"].values[0] != steering_type:
                continue
            question_group = group[group["question_id"]==717]

            assert len(question_group["activation_layer"].unique()) == 1
            assert len(question_group["activation_lambda"].unique()) == 1

            best_activation_layer = list(map(int, question_group["activation_layer"].unique()[0].split(',')))
            best_activation_lambda = question_group["activation_lambda"].unique()[0]
    
    del steered_data
    gc.collect()
    
    print(best_activation_layer, best_activation_lambda)
    
    run_activation_steering(steering_type, steering_model, test_questions, num_important_attributes, best_activation_layer=best_activation_layer, best_activation_lambda=best_activation_lambda)
    
    return steering_type, best_activation_layer, best_activation_lambda

def group_attribute_steering_test_questions(steering_model, num_important_attributes):
    database = Database()
    
    steering_type = f"activation_group_attribute_{num_important_attributes}"
    
    test_questions = database.get_steering_questions(split="test")
    steered_data = database.get_steering_generation_of_llm(llm_name=steering_model)

    best_activation_layer, best_activation_lambda = None, None
    for steering_type_index, (_, group) in enumerate(
        steered_data.groupby("steering_type")
    ):
        if group["steering_type"].values[0] != steering_type:
            continue
        question_group = group[group["question_id"]==717]
        
        assert len(question_group["activation_layer"].unique()) == 1
        assert len(question_group["activation_lambda"].unique()) == 1
        
        best_activation_layer = list(map(int, question_group["activation_layer"].unique()[0].split(',')))
        best_activation_lambda = question_group["activation_lambda"].unique()[0]
        
    del steered_data
    gc.collect()
    
    print(best_activation_layer, best_activation_lambda)
    
    run_activation_steering(steering_type, steering_model, test_questions, num_important_attributes, best_activation_layer=best_activation_layer, best_activation_lambda=best_activation_lambda)
    
    return steering_type, best_activation_layer, best_activation_lambda
    
    
def run_activation_steering(steering_type, steering_model, questions, num_important_attributes, best_activation_layer=None, best_activation_lambda=None):
    database = Database()
    
    if best_activation_layer != None:
        list_layer_indices = [best_activation_layer]
        lambdas = [best_activation_lambda]
        database.remove_steering_generation_of_llm_by_question_ids(steering_model, question_ids=questions["id"].values.tolist(), steering_type=steering_type)
    else:
        list_layer_indices = ACTIVATION_STEERING_INSERT_LAYER_INDECES[steering_model]
        lambdas = KONEN_ACTIVATION_SPACE_LAMBDAS
    print(list_layer_indices)
    print(lambdas)
    
    groups = database.get_groups(
        ["sfam_train", "sfam_val", "sfam_test", "lisa_train", "lisa_val", "lisa_test"]
    )
    
    steered_data = database.get_steering_generation_of_llm(steering_model, steering_type)
    steered_data = steered_data[steered_data["question_id"].isin(questions["id"].values.tolist())]
    
    if len(steered_data) < len(questions) * len(groups) * len(list_layer_indices) * len(lambdas):
        if len(steered_data) > 0:
            print(
                f"Activation steering is incomplete ({len(steered_data)} < {len(questions)} * {len(groups)} * {len(list_layer_indices)} * {len(lambdas)}); deleting data and recomputing"
            )
            database.remove_steering_generation_of_llm(steering_model, steering_type)
    else:
        print("Activation steering already done; skipping")
        return
    
    del steered_data
    gc.collect()
    
    #total_group_num = len(
    #    database.get_groups(
    #        [
    #            "sfam_train",
    #            "sfam_val",
    #            "sfam_test",
    #            "lisa_train",
    #            "lisa_val",
    #            "lisa_test",
    #        ]
    #    )
    #)
    #missing_groups = database.get_missing_activation_groups(
    #    ["sfam_train", "sfam_val", "sfam_test", "lisa_train", "lisa_val", "lisa_test"]
    #)
    #if len(missing_groups) == 0:
    #    print("All groups already steered; skipping activation steering")
    #    return
    
    
    
    answers = database.get_answers(
        ["sfam_train", "sfam_val", "sfam_test", "lisa_train", "lisa_val", "lisa_test"]
    )
    answers = answers[answers[STYLE_EMBEDDING_VECTOR].notnull()]
    
    activation_vectors_df = database.get_all_activation_vectors_of_llm(llm_name=steering_model)
    activation_vectors_df = activation_vectors_df.sort_values(
        ["style_vector_index", "layer"]
    )
    # make an np array with the shape (style_vector_index, layer, activation_vector)
    activation_vectors_df = activation_vectors_df.pivot(
        index="style_vector_index", columns="layer", values="activation_vector"
    )
    activation_vectors = activation_vectors_df.to_numpy()
    activation_vectors = np.array(
        [
            [activation_vectors[i][j] for j in range(activation_vectors.shape[1])]
            for i in range(activation_vectors.shape[0])
        ]
    )
    most_important_attribute_indeces = _get_most_important_attributes(
        np.array(answers[STYLE_EMBEDDING_VECTOR].values.tolist()),
        answers["data_group"].values,
    )
    style_vector_df = database.get_style_vector()
    nlp = spacy.load("en_core_web_sm")
    for group_index, group in enumerate(
#        missing_groups, start=total_group_num - len(missing_groups) + 1
        groups, start=len(groups)
    ):
        prompts = get_prompts(
            steering_type=steering_type,
            steering_model=steering_model,
            num_important_attributes=num_important_attributes,
            nlp=nlp,
            group=group,
            style_vector_df=style_vector_df,
            most_import_attributes=most_important_attribute_indeces[group][
                "dimensions"
            ],
            questions_df=questions,
        )
        group_answers = answers[answers["data_group"] == group]
        group_style_vector = np.mean(
            np.array(group_answers[STYLE_EMBEDDING_VECTOR].values.tolist()), axis=0
        )
        if USE_IMPORTANT_ATTRIBUTES_FOR_ACTIVATION_STEERING:
            group_dimensions = most_important_attribute_indeces[group]["dimensions"]
            sorted_style_vector_df = pd.DataFrame(
                style_vector_df.values[group_dimensions],
                style_vector_df.index[group_dimensions],
                style_vector_df.columns,
            )
            style_attribute_indeces = group_dimensions[
                sorted_style_vector_df["prompt_type"] == "Style prompt"
            ][:num_important_attributes]
            knowledge_attribute_indeces = group_dimensions[
                sorted_style_vector_df["prompt_type"] == "Knowledge prompt"
            ][:num_important_attributes]
            # group_style_vector softmax is zero where index not in style_attribute_indeces or knowledge_attribute_indeces
            allowed_indeces = np.isin(
                np.arange(len(group_style_vector)),
                np.concatenate((style_attribute_indeces, knowledge_attribute_indeces)),
            )
            group_style_vector[~allowed_indeces] = -np.inf
            
            ####################Debug
            group_style_vector_debug = pd.DataFrame(
                style_vector_df.values[allowed_indeces],
                style_vector_df.index[allowed_indeces],
                style_vector_df.columns,
            )
            style_attributes_df = group_style_vector_debug[
                group_style_vector_debug["prompt_type"] == "Style prompt"
            ].head(num_important_attributes)
            knowledge_attributes_df = group_style_vector_debug[
                group_style_vector_debug["prompt_type"] == "Knowledge prompt"
            ].head(num_important_attributes)
            print(f"Calculate activation vectors for {group} for the following indices:")
            print("style_attribute_indeces", style_attributes_df)
            print("knowledge_attribute_indeces", knowledge_attributes_df)
            ####################Debug
            
        group_softmax = np.exp(group_style_vector) / np.sum(np.exp(group_style_vector))
        print("group_softmax", group_softmax[group_softmax > 0])
        # weight the activation vectors at each layer by the softmax at the corresponding style vector index
        steering_vectors = np.tensordot(group_softmax, activation_vectors, axes=(0, 0))
        timer = time.time()
        steered_generation(
            steering_model=steering_model,
            steering_vectors=steering_vectors,
            prompt_dict=prompts,
            group=group,
            list_layer_indices=list_layer_indices,
            lambdas=lambdas
        )
        
    #print(database.get_steering_generation_of_llm(steering_model, steering_type)["answer"])

def extract_activation_vectors_difference(steering_model,
    layers: Optional[List[int]] = None,
):
    """
    Extract activation vectors from a model for given layers and a list of sentences.

    Args:
        model (PreTrainedModel): The model to extract activations from.
        tokenizer (PreTrainedTokenizer): The tokenizer to use for encoding the sentences.
        layers (Optional[List[int]]): The layers from which to extract activations. If None, all layers are used.
    """
    database = Database()
    style_vector = database.get_style_vector_missing_activation_vectors(steering_model)
    if style_vector.empty:
        print(
            "There are no style vectors missing activation vectors, skipping extraction."
        )
        return
    model = AutoModelForCausalLM.from_pretrained(steering_model, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(steering_model, device_map="auto")
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    result = []
    nlp = spacy.load("en_core_web_sm")
    timer = time.time()
    with torch.no_grad():
        #for index, vector_row in tqdm(
        #    style_vector.iterrows(),
        #    desc="Extracting activation vectors",
        #    total=len(style_vector),
        #    smoothing=0,
        #):
        for index, vector_row in style_vector.iterrows():
            if vector_row["prompt_type"] == "Knowledge prompt":
                sentence_pos = convert_attribute_sentence(
                nlp, vector_row["style_sentence"], perspective="I"
                )
                sentence_neg = convert_attribute_sentence(
                nlp, vector_row["style_sentence"], perspective="I_neg"
                )                
            else:
                sentence_pos = convert_attribute_sentence(
                nlp, vector_row["style_sentence"], perspective="you"
                )
                sentence_neg = convert_attribute_sentence(
                    nlp, vector_row["style_sentence"], perspective="you_neg"
                )
            print(sentence_pos)
            print(sentence_neg)
            sentence_prompt = re.sub(
                r"Cutting Knowledge Date:.*?\nToday Date:.*?\n+",  # match the two lines
                "",
                tokenizer.apply_chat_template(
                    [{"role": "system", "content": ""},
                     {"role": "user", "content": sentence_pos}],
                    add_generation_prompt=True,
                    tokenize=False,
                ),
                flags=re.S
            )

            tokenized_sentence = tokenizer(
                sentence_prompt,         
                return_tensors="pt",
            )

            if index == 0:
                print(vector_row["prompt_type"])
                raw_prompt = tokenizer.decode(tokenized_sentence["input_ids"][0], skip_special_tokens=False)
                print(raw_prompt)

            tokenized_sentence = {
                k: v.to(model.device) for k, v in tokenized_sentence.items()
            }
            outputs = model(**tokenized_sentence, output_hidden_states=True)

            # the first layer is the input embedding, so we skip it
            hidden_states_pos = outputs.hidden_states[1:]
            if layers is None:
                layers = range(len(hidden_states_pos))
            hidden_states_pos = [
                hidden_states_pos[i][0][-1].detach().cpu().numpy() for i in layers
            ]

            # Extract negative as in Turner et al., 2023?
            sentence_prompt = re.sub(
                    r"Cutting Knowledge Date:.*?\nToday Date:.*?\n+",  # match the two lines
                    "",
                    tokenizer.apply_chat_template(
                        [{"role": "system", "content": ""},
                         {"role": "user", "content": sentence_neg}],
                        add_generation_prompt=True,
                        tokenize=False,
                    ),
                    flags=re.S
                )

            tokenized_sentence = tokenizer(
                sentence_prompt,         
                return_tensors="pt",
            )

            if index == 0:
                raw_prompt = tokenizer.decode(tokenized_sentence["input_ids"][0], skip_special_tokens=False)
                print(raw_prompt)

            tokenized_sentence = {
                k: v.to(model.device) for k, v in tokenized_sentence.items()
            }
            outputs = model(**tokenized_sentence, output_hidden_states=True)

            # the first layer is the input embedding, so we skip it
            hidden_states_neg = outputs.hidden_states[1:]
            if layers is None:
                layers = range(len(hidden_states_neg))
            hidden_states_neg = [
                hidden_states_neg[i][0][-1].detach().cpu().numpy() for i in layers
            ]

            hidden_states = [
                pos_activation - neg_activation
                for pos_activation, neg_activation in zip(hidden_states_pos, hidden_states_neg)
            ]
            
            result.extend(
                [
                    [vector_row["style_vector_index"], steering_model, hs, layer]
                    for hs, layer in zip(hidden_states, layers)
                ]
            )
    log_time("Extracting activation vectors", timer, logger)
    result = pd.DataFrame(
        result, columns=["style_vector_index", "llm_name", "activation_vector", "layer"]
    )
    database.insert_activation_vectors(result)
    del model
    torch.cuda.empty_cache()
    #torch.distributed.destroy_process_group()
    gc.collect()

class RepetitionStoppingCriteria(StoppingCriteria):
    def __init__(self, n, repetition_threshold):
        """
        Custom stopping criteria to stop generation when an n-gram is repeated more than a threshold.
        
        Args:
            n (int): The size of the n-gram to track (e.g., 2 for bigrams, 3 for trigrams).
            repetition_threshold (int): How many times an n-gram can appear before stopping.
        """
        self.n = n
        self.repetition_threshold = repetition_threshold

    def generate_ngrams(self, tokens):
        output = []  
        for i in range(len(tokens)- self.n+1):
            output.append(tokens[i:i+self.n])
        return output
        
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs):
        # Convert input IDs to list of tokens 
        generated_tokens = input_ids[0].tolist()
        
        # Only start checking once we have enough tokens for an n-gram
        if len(generated_tokens) < self.n:
            return False

        generated_ngrams = self.generate_ngrams(generated_tokens)
        
        # Get the last n-gram
        ngram = generated_ngrams[-1]

        # Stop if the last n-gram has been repeated more than the threshold
        if generated_ngrams.count(ngram) > self.repetition_threshold:
            return True

        return False

def steered_generation(
    steering_model: str,
    steering_vectors: np.ndarray,
    prompt_dict: [str, Dict[str, Union[str, BatchEncoding, int]]],
    group: str,
    list_layer_indices,
    lambdas
) -> str:
    database = Database()
    
    batch_size = 100
    
    tokenizer = AutoTokenizer.from_pretrained(steering_model, device_map="auto")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Define the custom stopping criteria: stop if any ngram repeats more than 5 times
    #repetition_stopping_criteria = RepetitionStoppingCriteria(n=10, repetition_threshold=5)
    
    with torch.no_grad():
        for layer_indeces in tqdm(
            list_layer_indices,
            desc="Steering activation layers",
            smoothing=0,
        ):
            layer_timer = time.time()
            if not isinstance(layer_indeces, list):
                layer_indeces = [layer_indeces]
            model = AutoModelForCausalLM.from_pretrained(
                steering_model,
                dtype=torch.bfloat16,
                #device_map="cuda",
                attn_implementation="flash_attention_2"
            )
            
            ds_engine = deepspeed.init_inference(
                model=model,
                dtype=torch.bfloat16,
                replace_with_kernel_inject=False,
            )
            model = ds_engine.module
            model.eval()
            
            steering_layers = []
            for layer_index in layer_indeces:
                model.model.layers[layer_index].mlp = SteeringLayer(
                    model.model.layers[layer_index].mlp,
                    torch.from_numpy(steering_vectors[layer_index])
                    .unsqueeze(0)
                    .to(model.device)
                    .to(torch.bfloat16),
                )
                steering_layers.append(model.model.layers[layer_index].mlp)
            steered_data = []
            for lmbda in lambdas:
                for steering_layer in steering_layers:
                    steering_layer.lmbda = lmbda
                for steering_type, prompt_data in prompt_dict.items():
                    answers = []
                    for batch in _chunk_batch_encoding(
                        prompt_data["encoded"], batch_size=batch_size
                    ):
                        output = model.generate(
                            **batch,
                            do_sample=False,
                            max_new_tokens=700,
                            temperature=None,
                            top_p=None,
                            #stopping_criteria=StoppingCriteriaList([repetition_stopping_criteria])
                            
                        )
                        
                        # Remove prompt from generated output
                        output = output[:, prompt_data["encoded"]["input_ids"].shape[-1]:]
                        decoded_output = tokenizer.batch_decode(output, skip_special_tokens=True)

                        answers.extend(
                            decoded_output
                        )
                    #answers = [
                    #    re.sub(
                    #        r"^.*?assistant", "", answer, flags=re.DOTALL, count=1
                    #    ).strip()
                    #    for answer in answers
                    #]
                    #answers = [
                    #    re.sub(r"^[\s]*?assistant", "", answer, flags=re.DOTALL).strip()
                    #    for answer in answers
                    #]
                    steered_data.append(
                        pd.DataFrame(
                            {
                                "question_id": prompt_data["id"],
                                "data_group": group,
                                "steering_type": steering_type,
                                "llm_name": steering_model,
                                "answer": answers,
                                "prompt": prompt_data["raw"],
                                "activation_layer": ",".join(
                                    str(i) for i in layer_indeces
                                ),
                                "activation_vector": steering_vectors[
                                    layer_indeces
                                ].tobytes(),
                                "activation_lambda": lmbda,
                            }
                        )
                    )
            print(pd.concat(steered_data)[["answer", "activation_layer", "activation_lambda"]])
            database.insert_steering_generation(pd.concat(steered_data))
            del steering_layers
            del ds_engine
            del model
            torch.cuda.empty_cache()
            #torch.distributed.destroy_process_group()
            gc.collect()
            layer_indeces = ", ".join(str(i) for i in layer_indeces)
            log_time(
                f"Activation steering generation for group {group} and layer {layer_indeces}",
                layer_timer,
                logger,
                "debug",
            )


def get_prompts(
    steering_type: str,
    steering_model: str,
    num_important_attributes: int,
    nlp: spacy.language.Language,
    group: str,
    style_vector_df: pd.DataFrame,
    most_import_attributes: np.ndarray,
    questions_df: pd.DataFrame,
) -> Dict[str, Dict[str, Union[str, BatchEncoding, int]]]:
    tokenizer = AutoTokenizer.from_pretrained(steering_model, device_map="auto")
    tokenizer.pad_token = tokenizer.eos_token
    result = {}
    system_prompt = None
    if steering_type == f"activation_group_{num_important_attributes}":
        system_prompt = get_system_prompt_group(group)
    elif steering_type == f"activation_konen_group":
        system_prompt = get_system_prompt_group(group)
    elif steering_type == f"activation_group_attribute_{num_important_attributes}":
            system_prompt = get_system_prompt_group_attribute(
                num_important_attributes,
                nlp,
                group,
                style_vector_df,
                most_import_attributes,
            )
    elif steering_type == f"activation_konen_group_attribute_{num_important_attributes}":
            system_prompt = get_system_prompt_group_attribute(
                num_important_attributes,
                nlp,
                group,
                style_vector_df,
                most_import_attributes,
            )
            
    if system_prompt is None:
        print(
            f"System prompt for group {group} is None, skipping activation steering."
        )
        return {
            key: []
            for key in [
                "activation_group",
                "activation_group_attribute",
            ]
        }
    
    prompts = [
         re.sub(
            r"Cutting Knowledge Date:.*?\nToday Date:.*?\n+",  # match the two lines
            "",
            tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": x},
                ],
                add_generation_prompt=True,
                tokenize=False,
            ),
            flags=re.S
        )
        for x in questions_df["question"].to_list()
    ]

    #print(prompts[0])
    #time.sleep(10)

    prompts_encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        padding_side="left"
    ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    
    result[steering_type] = {
        "id": questions_df["id"].to_list(),
        "raw": prompts,
        "encoded": prompts_encoded,
    }
    return result


def _chunk_batch_encoding(encoded_inputs, batch_size):
    # Convert to list of dicts (one per item in the batch)
    total = encoded_inputs.input_ids.shape[0]

    for i in range(0, total, batch_size):
        chunk = {k: v[i : i + batch_size] for k, v in encoded_inputs.items()}
        yield BatchEncoding(chunk)
