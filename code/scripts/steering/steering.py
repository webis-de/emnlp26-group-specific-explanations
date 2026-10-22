import json
import logging
import random
import pandas as pd

from config_new_prompts import (
    MODEL_OUTPUT_FOLDER,
    STYLE_EMBEDDING_VECTOR,
)
from scripts import embedder, lisa, sfam, normal_embedder
from scripts.steering.activation_steering import activation_steering
from scripts.steering.activation_steering_konen import konen_activation_steering
from scripts.evaluate import evaluate_all
from utils.database import Database
from utils.llama import Llama
from . import prompt_steering
import gc
import torch

logger = logging.getLogger(__name__)


def get_attributes():
    database = Database()
    groups = database.get_groups(
        ["sfam_train", "sfam_val", "sfam_test", "lisa_train", "lisa_val", "lisa_test"]
    )
    return prompt_steering.get_attributes_per_group(group_names=groups)

def steer_test(steering_model, num_important_attributes, recompute=False):
    database = Database()
    
    llama = Llama(steering_model)
    
    # Prompt-based steering
    # Test data split
    questions = database.get_steering_questions(split="rooein_test")
    
    prompt_steering.group_test_steering(llama, questions=questions, group_names=["gamedev"])
    
    del llama
    torch.cuda.empty_cache()
    gc.collect()

def steer_test_questions(steering_model, num_important_attributes, recompute=False):
    database = Database()
    
    if recompute:
        database.remove_steering_generation()
        database.delete_activation_vectors()
    
    llama = Llama(steering_model)
    groups = database.get_groups(
        ["sfam_train", "sfam_val", "sfam_test", "lisa_train", "lisa_val", "lisa_test"]
    )
    import_questions()
    
    current_steered_data = database.get_steering_generation_of_llm(steering_model)
    steering_types = current_steered_data["steering_type"].unique()
    print("Steering types:", steering_types)
    
    del current_steered_data
    gc.collect()
    
    # Prompt-based steering
    # Test data split
    questions = database.get_steering_questions(split="test")
    
    if "base" in steering_types:
        print("recompute base")
        database.remove_steering_generation_of_llm_by_question_ids(steering_model, question_ids=questions["id"].values.tolist(), steering_type="base")
        prompt_steering.base_steering(llama, questions=questions)

    if "prompt_group" in steering_types:
        print("recompute prompt_group")
        database.remove_steering_generation_of_llm_by_question_ids(steering_model, question_ids=questions["id"].values.tolist(), steering_type="prompt_group")
        prompt_steering.group_steering(llama, questions=questions, group_names=groups)
    
    if f"prompt_group_attribute_{num_important_attributes}" in steering_types:
        print(f"recompute prompt_group_attribute_{num_important_attributes}")
        database.remove_steering_generation_of_llm_by_question_ids(steering_model, question_ids=questions["id"].values.tolist(), steering_type=f"prompt_group_attribute_{num_important_attributes}")
        prompt_steering.group_attribute_steering(
            llama, questions=questions, group_names=groups, num_important_attributes=num_important_attributes
        )
    
    del llama
    torch.cuda.empty_cache()
    gc.collect()
    
    # Extract activation vectors
    konen_activation_steering.extract_activation_vectors(steering_model)
    activation_steering.extract_activation_vectors_difference(steering_model)
    
    best_activation_hyperparameters = {}
    
    # Activation-based steering (Konen et al., 2024)
    if f"activation_konen_group" in steering_types:
        print(f"recompute activation_konen_group")
        steering_type, best_activation_layer, best_activation_lambda = konen_activation_steering.group_steering_test_questions(steering_model, num_important_attributes)
        best_activation_hyperparameters[steering_type] = {"layer_indices": best_activation_layer, "lambda": best_activation_lambda}
    
    if f"activation_konen_group_attribute_{num_important_attributes}" in steering_types:
        print(f"recompute activation_konen_group_attribute_{num_important_attributes}")
        steering_type, best_activation_layer, best_activation_lambda = konen_activation_steering.group_attribute_steering_test_questions(steering_model, num_important_attributes)
        best_activation_hyperparameters[steering_type] = {"layer_indices": best_activation_layer, "lambda": best_activation_lambda}
    
    # Attribute activation-based steering (Our approach)
    if f"activation_group_{num_important_attributes}" in steering_types:
        print(f"recompute activation_group_{num_important_attributes}")
        steering_type, best_activation_layer, best_activation_lambda = activation_steering.group_steering_test_questions(steering_model, num_important_attributes)
        best_activation_hyperparameters[steering_type] = {"layer_indices": best_activation_layer, "lambda": best_activation_lambda}
    
    if f"activation_group_attribute_{num_important_attributes}" in steering_types:
        print(f"recompute activation_group_attribute_{num_important_attributes}")
        steering_type, best_activation_layer, best_activation_lambda = activation_steering.group_attribute_steering_test_questions(steering_model, num_important_attributes)
        best_activation_hyperparameters[steering_type] = {"layer_indices": best_activation_layer, "lambda": best_activation_lambda}
    
    _create_style_vectors(steering_model)
    _create_style_embeddings(steering_model)
    _create_normal_embeddings(steering_model)
    evaluate_all(steering_model)
    
    cursor = database._conn.cursor()
    cursor.execute("VACUUM;")
    database._conn.commit()
    cursor.close()
    
    print(best_activation_hyperparameters)
    #database.backup("steering")

def steer_act_diff(steering_model, num_important_attributes, recompute=False):
    database = Database()
    
    if recompute:
        database.remove_steering_generation()
        database.delete_activation_vectors()
    
    groups = database.get_groups(
        ["sfam_train", "sfam_val", "sfam_test", "lisa_train", "lisa_val", "lisa_test"]
    )
    import_questions()
    
    current_steered_data = database.get_steering_generation_of_llm(steering_model)
    steering_types = current_steered_data["steering_type"].unique()
    print("Steering types:", steering_types)
    
    del current_steered_data
    gc.collect()
    
    
    best_activation_hyperparameters = {}
    
    if database.file_name == "qwen_data_qwen_filtered_sent.sqlite":
        # Activation-based steering (Konen et al., 2024)
        if f"activation_konen_group" in steering_types:
            print(f"recompute activation_konen_group")
            steering_type, best_activation_layer, best_activation_lambda = konen_activation_steering.group_steering_test_questions(steering_model, num_important_attributes)
            best_activation_hyperparameters[steering_type] = {"layer_indices": best_activation_layer, "lambda": best_activation_lambda}
    if database.file_name == "qwen_data_qwen_filtered_sent.sqlite":
        if f"activation_konen_group_attribute_{num_important_attributes}" in steering_types:
            print(f"recompute activation_konen_group_attribute_{num_important_attributes}")
            steering_type, best_activation_layer, best_activation_lambda = konen_activation_steering.group_attribute_steering_test_questions(steering_model, num_important_attributes)
            best_activation_hyperparameters[steering_type] = {"layer_indices": best_activation_layer, "lambda": best_activation_lambda}
    
    # Attribute activation-based steering (Our approach)
    if f"activation_group_{num_important_attributes}" in steering_types:
        print(f"recompute activation_group_{num_important_attributes}")
        steering_type, best_activation_layer, best_activation_lambda = activation_steering.group_steering_test_questions(steering_model, num_important_attributes)
        best_activation_hyperparameters[steering_type] = {"layer_indices": best_activation_layer, "lambda": best_activation_lambda}
    
    if f"activation_group_attribute_{num_important_attributes}" in steering_types:
        print(f"recompute activation_group_attribute_{num_important_attributes}")
        steering_type, best_activation_layer, best_activation_lambda = activation_steering.group_attribute_steering_test_questions(steering_model, num_important_attributes)
        best_activation_hyperparameters[steering_type] = {"layer_indices": best_activation_layer, "lambda": best_activation_lambda}
    
    _create_style_vectors(steering_model)
    _create_style_embeddings(steering_model)
    _create_normal_embeddings(steering_model)
    evaluate_all(steering_model)
    
    cursor = database._conn.cursor()
    cursor.execute("VACUUM;")
    database._conn.commit()
    cursor.close()
    
    print(best_activation_hyperparameters)
    #database.backup("steering")

def steer_3_att_10_att(steering_model, num_important_attributes, recompute=False):
    database = Database()
    
    if recompute:
        database.remove_steering_generation()
        database.delete_activation_vectors()
    
    llama = Llama(steering_model)
    groups = database.get_groups(
        ["sfam_train", "sfam_val", "sfam_test", "lisa_train", "lisa_val", "lisa_test"]
    )
    import_questions()
    
    # Prompt-based steering
    # Validation data split
    val_questions = database.get_steering_questions(split="val")
    #val_questions = val_questions[:10]
    
    prompt_steering.base_steering(llama, questions=val_questions)
    
    # Test data split
    questions = database.get_steering_questions(split="test")
    #questions = pd.concat([database.get_steering_questions(split="eli5_test")[:20], database.get_steering_questions(split="rooein_test")[:20]], ignore_index=True)
    
    prompt_steering.base_steering(llama, questions=questions)
    prompt_steering.group_attribute_steering(
        llama, questions=questions, group_names=groups, num_important_attributes=num_important_attributes
    )
    del llama
    torch.cuda.empty_cache()
    gc.collect()
    
    # Extract activation vectors
    #konen_activation_steering.extract_activation_vectors(steering_model)
    activation_steering.extract_activation_vectors(steering_model)
    #activation_steering.extract_activation_vectors_difference(steering_model) # only for data_qwen_filtered_sent_new_new_act_diff
    
    best_activation_hyperparameters = {}
    
    # Activation-based steering (Konen et al., 2024)
    #steering_type, best_activation_layer, best_activation_lambda = konen_activation_steering.group_attribute_steering(steering_model, num_important_attributes)
    #best_activation_hyperparameters[steering_type] = {"layer_indices": best_activation_layer, "lambda": best_activation_lambda}
    
    # Attribute activation-based steering (Our approach)
    #steering_type, best_activation_layer, best_activation_lambda = activation_steering.group_steering(steering_model, num_important_attributes)
    #best_activation_hyperparameters[steering_type] = {"layer_indices": best_activation_layer, "lambda": best_activation_lambda}
    steering_type, best_activation_layer, best_activation_lambda = activation_steering.group_attribute_steering(steering_model, num_important_attributes)
    best_activation_hyperparameters[steering_type] = {"layer_indices": best_activation_layer, "lambda": best_activation_lambda}
    
    print(best_activation_hyperparameters)
    
    _create_style_vectors(steering_model)
    _create_style_embeddings(steering_model)
    _create_normal_embeddings(steering_model)
    evaluate_all(steering_model)
    #database.backup("steering")
    
def steer_big(steering_model, num_important_attributes, recompute=False):
    database = Database()
    
    llama = Llama(steering_model)
    groups = database.get_groups(
        ["sfam_train", "sfam_val", "sfam_test", "lisa_train", "lisa_val", "lisa_test"]
    )
    import_questions()
    
    # Prompt-based steering
    # Test data split
    questions = database.get_steering_questions(split="test")
    
    print(f"recompute prompt_group_attribute_{num_important_attributes}")
    database.remove_steering_generation_of_llm_by_question_ids(steering_model, question_ids=questions["id"].values.tolist(), steering_type=f"prompt_group_attribute_{num_important_attributes}")
    prompt_steering.group_attribute_steering(
        llama, questions=questions, group_names=groups, num_important_attributes=num_important_attributes
    )
    
    del llama
    torch.cuda.empty_cache()
    gc.collect()
    
    _create_style_vectors(steering_model)
    _create_style_embeddings(steering_model)
    _create_normal_embeddings(steering_model)
    evaluate_all(steering_model)
    
    cursor = database._conn.cursor()
    cursor.execute("VACUUM;")
    database._conn.commit()
    cursor.close()

def steer(steering_model, num_important_attributes, recompute=False):
    #activation_steering.extract_activation_vectors_difference(steering_model)
    #exit()
    #database = Database()
    
    
    #llama = Llama(steering_model)
    #import_questions()
    
    # Validation data split
    #val_questions = database.get_steering_questions(split="val")
    #val_questions = val_questions[:10]
    
    #prompt_steering.base_steering(llama, questions=val_questions)
    
    # Test data split
    #questions = database.get_steering_questions(split="test")
    #questions = pd.concat([database.get_steering_questions(split="eli5_test")[:20], database.get_steering_questions(split="rooein_test")[:20]], ignore_index=True)
    
    #prompt_steering.base_steering(llama, questions=questions)
    
    #konen_activation_steering.extract_activation_vectors(steering_model)
    
    #best_activation_hyperparameters = {}
    #steering_type, best_activation_layer, best_activation_lambda = konen_activation_steering.group_steering(steering_model, num_important_attributes)
    #best_activation_hyperparameters[steering_type] = {"layer_indices": best_activation_layer, "lambda": best_activation_lambda}
    #steering_type, best_activation_layer, best_activation_lambda = konen_activation_steering.group_attribute_steering(steering_model, num_important_attributes)
    #best_activation_hyperparameters[steering_type] = {"layer_indices": best_activation_layer, "lambda": best_activation_lambda}
    #print(best_activation_hyperparameters)
    
    #_create_style_vectors(steering_model)
    #_create_style_embeddings(steering_model)
    #_create_normal_embeddings(steering_model)
    #exit()
    
    database = Database()
    
    if recompute:
        database.remove_steering_generation()
        database.delete_activation_vectors()
    
    llama = Llama(steering_model)
    groups = database.get_groups(
        ["sfam_train", "sfam_val", "sfam_test", "lisa_train", "lisa_val", "lisa_test"]
    )
    import_questions()
    
    # Prompt-based steering
    # Validation data split
    val_questions = database.get_steering_questions(split="val")
    #val_questions = val_questions[:10]
    
    prompt_steering.base_steering(llama, questions=val_questions)
    
    # Test data split
    questions = database.get_steering_questions(split="test")
    #questions = pd.concat([database.get_steering_questions(split="eli5_test")[:20], database.get_steering_questions(split="rooein_test")[:20]], ignore_index=True)
    
    prompt_steering.base_steering(llama, questions=questions)
    prompt_steering.group_steering(llama, questions=questions, group_names=groups)
    prompt_steering.group_attribute_steering(
        llama, questions=questions, group_names=groups, num_important_attributes=num_important_attributes
    )
    del llama
    torch.cuda.empty_cache()
    gc.collect()
    
    # Extract activation vectors
    konen_activation_steering.extract_activation_vectors(steering_model)
    activation_steering.extract_activation_vectors(steering_model)
    #activation_steering.extract_activation_vectors_difference(steering_model) # only for data_qwen_filtered_sent_new_new_act_diff
    
    best_activation_hyperparameters = {}
    
    # Activation-based steering (Konen et al., 2024)
    steering_type, best_activation_layer, best_activation_lambda = konen_activation_steering.group_steering(steering_model, num_important_attributes)
    best_activation_hyperparameters[steering_type] = {"layer_indices": best_activation_layer, "lambda": best_activation_lambda}
    steering_type, best_activation_layer, best_activation_lambda = konen_activation_steering.group_attribute_steering(steering_model, num_important_attributes)
    best_activation_hyperparameters[steering_type] = {"layer_indices": best_activation_layer, "lambda": best_activation_lambda}
    
    # Attribute activation-based steering (Our approach)
    steering_type, best_activation_layer, best_activation_lambda = activation_steering.group_steering(steering_model, num_important_attributes)
    best_activation_hyperparameters[steering_type] = {"layer_indices": best_activation_layer, "lambda": best_activation_lambda}
    steering_type, best_activation_layer, best_activation_lambda = activation_steering.group_attribute_steering(steering_model, num_important_attributes)
    best_activation_hyperparameters[steering_type] = {"layer_indices": best_activation_layer, "lambda": best_activation_lambda}
    
    print(best_activation_hyperparameters)
    
    _create_style_vectors(steering_model)
    _create_style_embeddings(steering_model)
    _create_normal_embeddings(steering_model)
    evaluate_all(steering_model)
    #database.backup("steering")


def _create_style_vectors(steering_model, steered_data=[]):
    database = Database()
    if len(steered_data) == 0:
        steered_data = database.get_steering_generation_of_llm(llm_name=steering_model)
    steered_data = steered_data[steered_data["style_vector"].isnull()]
    if len(steered_data) == 0:
        print("Style vectors already created; skipping")
        return

    #if STYLE_EMBEDDING_VECTOR == "style_vector_lisa":
    style_vectors = lisa.create_style_vector(
        f"{MODEL_OUTPUT_FOLDER}/lisa", steered_data
    )
    style_vectors = style_vectors.rename(
        columns={"answer_id": "id", "style_vector": "vector"}
    )
    database.update_steering_generations_vector(style_vectors, "style_vector")
    #elif STYLE_EMBEDDING_VECTOR == "style_vector_sfam":
#
    #    def save_func(result):
    #        result = result.rename(
    #            columns={"answer_id": "id", "style_vector": "vector"}
    #        )
    #        database.update_steering_generations_vector(result, "style_vector")
#
    #    sfam.create_style_vector(f"{MODEL_OUTPUT_FOLDER}/sfam", steered_data, save_func)
    #else:
    #    logger.error(f"Unknown style embedding vector type: {STYLE_EMBEDDING_VECTOR}")
    #    raise ValueError(
    #        f"Unknown style embedding vector type: {STYLE_EMBEDDING_VECTOR}"
    #    )


def _create_style_embeddings(steering_model, steered_data=[]):
    database = Database()
    if len(steered_data) == 0:
        steered_data = database.get_steering_generation_of_llm(llm_name=steering_model)
    steered_data = steered_data[steered_data["style_embedding"].isnull()]
    if len(steered_data) == 0:
        print("Style embeddings already created; skipping")
        return
    
    embeddings = embedder.create_style_embedding(
        f"{MODEL_OUTPUT_FOLDER}/embedder", steered_data
    )
    embeddings = embeddings.rename(
        columns={"answer_id": "id", "style_embedding": "vector"}
    )
    database.update_steering_generations_vector(embeddings, "style_embedding")

def _create_normal_embeddings(steering_model, steered_data=[]):
    database = Database()
    if len(steered_data) == 0:
        steered_data = database.get_steering_generation_of_llm(llm_name=steering_model)
    steered_data = steered_data[steered_data["normal_embedding"].isnull()]
    if len(steered_data) == 0:
        print("Normal embeddings already created; skipping")
        return
    
    normal_embedder.get_normal_embedding(steered_data)
    
    
def import_questions():
    database = Database()

    with open("data/kilt_eli5_val_questions.json", "r") as f:
        eli5_val = json.load(f)
    eli5_val = [x["input"] for x in eli5_val]
    
    with open("data/kilt_eli5_test_questions.json", "r") as f:
        eli5_test = json.load(f)
    eli5_test = [x["input"] for x in eli5_test]
    with open("data/rooein_et_al_2023_questions.json", "r") as f:
        rooein = json.load(f)
    rooein = [x["input"] for x in rooein]
    questions = pd.concat(
        [
            pd.DataFrame({"question": eli5_val, "source": "eli5_val"}),
            pd.DataFrame({"question": eli5_test, "source": "eli5_test"}),
            pd.DataFrame({"question": rooein, "source": "rooein_test"}),
        ]
    )
    existing_questions = database.get_steering_questions()
    questions = questions[~questions["question"].isin(existing_questions["question"])]
    if questions.empty:
        print("There are no new steering questions; skipping import")
        return
    else:
        print("Imported steering questions")
    database.insert_steering_questions(questions)
