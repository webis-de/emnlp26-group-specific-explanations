import gc
import time
from typing import Dict, List, Optional, Union
import pandas as pd
import spacy
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BatchEncoding
import torch
import logging
import numpy as np
import re

from config_new_prompts import (
    KONEN_ACTIVATION_SPACE_LAMBDAS,
    NUM_IMPORTANT_ATTRIBUTES_KNOWLEDGE,
    NUM_IMPORTANT_ATTRIBUTES_STYLE,
    STYLE_EMBEDDING_VECTOR,
    USE_IMPORTANT_ATTRIBUTES_FOR_ACTIVATION_STEERING,
    ACTIVATION_STEERING_INSERT_LAYER_INDECES,
)
from helper import log_time
from scripts.steering.activation_steering.steering_layer import SteeringLayer
from scripts.steering.activation_steering.activation_steering import evaluate, steered_generation, get_prompts, _chunk_batch_encoding
from scripts.steering.prompt_steering import (
    _get_most_important_attributes,
    convert_attribute_sentence,
    get_system_prompt_attribute,
    get_system_prompt_group,
    get_system_prompt_group_attribute
    
)
from scripts.evaluate import _evaluate_activation_steering_joint_score
from utils.database import Database

import deepspeed

logger = logging.getLogger(__name__)


def group_steering(steering_model, num_important_attributes):
    database = Database()
    
    steering_type = f"activation_konen_group"
    
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
    
    steering_type = f"activation_konen_group_attribute_{num_important_attributes}"
    
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
    
    steering_type = f"activation_konen_group"
    
    test_questions = database.get_steering_questions(split="test")
    steered_data = database.get_steering_generation_of_llm(llm_name=steering_model)

    best_activation_layer, best_activation_lambda = None, None
    
    if database.file_name == "qwen_data_qwen_filtered_sent.sqlite":
        best_activation_layer = [17,18,19]
        best_activation_lambda = 1.75
    elif database.file_name == "llama_data_qwen_filtered_sent.sqlite":
        best_activation_layer = [15,16,17]
        best_activation_lambda = 1.25
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
    
    steering_type = f"activation_konen_group_attribute_{num_important_attributes}"
    
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
                f"Activation steering konen is incomplete ({len(steered_data)} < {len(questions)} * {len(groups)} * {len(list_layer_indices)} * {len(lambdas)}); deleting data and recomputing"
            )
            database.remove_steering_generation_of_llm(steering_model, steering_type)
    else:
        print("Activation steering konen already done; skipping")
        return  
    
    del steered_data
    gc.collect()
    
    answers = database.get_answers(
        ["sfam_train", "sfam_val", "sfam_test", "lisa_train", "lisa_val", "lisa_test"]
    )
    answers = answers[answers[STYLE_EMBEDDING_VECTOR].notnull()]
    answer_ids = answers["id"].values.tolist()
    
    activation_vectors_df = database.get_all_activation_vectors_konen_of_llm(answer_ids=answer_ids, llm_name=steering_model)
    
    #activation_vectors_df = activation_vectors_df.sort_values(
    #    ["answer_id", "layer"]
    #)
    # make an np array with the shape (answer_id, layer, activation_vector)
    #activation_vectors_df = activation_vectors_df.pivot(
    #    index="style_vector_index", columns="layer", values="activation_vector"
    #)
    #activation_vectors = activation_vectors_df.to_numpy()
    #activation_vectors = np.array(
    #    [
    #        [activation_vectors[i][j] for j in range(activation_vectors.shape[1])]
    #        for i in range(activation_vectors.shape[0])
    #    ]
    #)
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
        # Extract activations vectors based on answers of current group
        group_activation_vectors = activation_vectors_df[activation_vectors_df["data_group"] == group]
        group_activation_vectors = group_activation_vectors.pivot(
            index="answer_id", columns="layer", values="activation_vector"
        )
        group_activation_vectors = group_activation_vectors.to_numpy()
        group_activation_vectors = np.array(
            [
                [group_activation_vectors[i][j] for j in range(group_activation_vectors.shape[1])]
                for i in range(group_activation_vectors.shape[0])
            ]
        )
        group_activation_vectors_mean = np.mean(group_activation_vectors, axis=0)
        
        # Extract activations vectors based on answers of all other groups
        all_other_group_activation_vectors = activation_vectors_df[activation_vectors_df["data_group"] != group]
        all_other_group_activation_vectors = all_other_group_activation_vectors.pivot(
            index="answer_id", columns="layer", values="activation_vector"
        )
        all_other_group_activation_vectors = all_other_group_activation_vectors.to_numpy()
        all_other_group_activation_vectors = np.array(
            [
                [all_other_group_activation_vectors[i][j] for j in range(all_other_group_activation_vectors.shape[1])]
                for i in range(all_other_group_activation_vectors.shape[0])
            ]
        )
        all_other_group_activation_vectors_mean = np.mean(all_other_group_activation_vectors, axis=0)
        
        # Konen et al., 2024, equation 5
        steering_vectors = group_activation_vectors_mean - all_other_group_activation_vectors_mean
        
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


def extract_activation_vectors(steering_model,
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
    answers = database.get_answers()
    answers_ids = answers[answers[STYLE_EMBEDDING_VECTOR].notnull()]["id"].values.tolist()
    
    answers = database.get_answers_missing_activation_vectors_konen(answers_ids, steering_model)
    if answers.empty:
        print(
            "There are no answers missing activation vectors, skipping extraction."
        )
        return
    model = AutoModelForCausalLM.from_pretrained(steering_model, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(steering_model, device_map="auto")
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    result = []
    timer = time.time()
    with torch.no_grad():
        for index, answer_row in tqdm(
            answers.iterrows(),
            desc="Extracting activation vectors",
            total=len(answers),
            smoothing=0,
        ):
            answer = answer_row["answer"]
            
            input_tokens = tokenizer(answer, return_tensors="pt").to(model.device)
        
            outputs = model.forward(input_tokens.input_ids, output_hidden_states=True)
            # the first layer is the input embedding, so we skip it
            hidden_states = outputs.hidden_states[1:]
            
            if layers is None:
                layers = range(len(hidden_states))
            hidden_states = [
                hidden_states[i][0][-1].detach().cpu().numpy() for i in layers
            ]
            
            result.extend(
                [
                    [answer_row["id"], steering_model, hs, layer]
                    for hs, layer in zip(hidden_states, layers)
                ]
            )
    log_time("Extracting activation vectors", timer, logger)
    result = pd.DataFrame(
        result, columns=["answer_id", "llm_name", "activation_vector", "layer"]
    )
    database.insert_activation_vectors_konen(result)
    del model
    torch.cuda.empty_cache()
    #torch.distributed.destroy_process_group()
    gc.collect()