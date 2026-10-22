from typing import Dict, List, Literal, Tuple, Union
import pandas as pd
import logging
import re
import spacy
import numpy as np
from scipy.stats import pointbiserialr
import time

from config_new_prompts import (
    GROUP_STEERING_NAMES,
    STYLE_EMBEDDING_VECTOR,
    NUM_IMPORTANT_ATTRIBUTES_STYLE,
    NUM_IMPORTANT_ATTRIBUTES_KNOWLEDGE,
    NUM_IMPORTANT_ATTRIBUTES
)
from data import prompts
from helper import log_time
from utils.database import Database
from utils.llama import Llama

logger = logging.getLogger(__name__)

def group_test_steering(llama: Llama, questions: pd.DataFrame, group_names: List[str]):
    database = Database()

    our_prompt = []
    for question in questions["question"].tolist():
        if "digestion" in question:
            our_prompt.append(question)
    reviewer_prompt = ["What is the role of enzymes in digestion? Explain in a way understandable for game developers.",
                      "What is the role of enzymes in digestion? Explain in a way understandable for game developers. Your explanations have to be shorter than 200 words."]
    
    for group_name in group_names:
        system_prompt = get_system_prompt_group(group_name)
        if system_prompt is None:
            logger.error(
                f"Skipping group steering for group {group_name} as it is not in GROUP_STEERING_NAMES"
            )
            continue
        
        answers, llama_prompts = llama.batch_generate(
            our_prompt,
            random_seed=42,
            system_prompt=system_prompt.replace("Your explanations have to be shorter than 200 words.", ""),
            return_prompts=True,
        )
        print("Our prompting baseline (prompt_group)")
        for llama_prompt, answer in zip(llama_prompts, answers):
            print(llama_prompt)
            print(answer)
        
        answers, llama_prompts = llama.batch_generate(
            our_prompt,
            random_seed=42,
            system_prompt=system_prompt,
            return_prompts=True,
        )
        for llama_prompt, answer in zip(llama_prompts, answers):
            print(llama_prompt)
            print(answer)
        
        answers, llama_prompts = llama.batch_generate(
            reviewer_prompt,
            random_seed=42,
            system_prompt=None,
            return_prompts=True,
        )
        print("Prompting baseline of reviewer")
        for llama_prompt, answer in zip(llama_prompts, answers):
            print(llama_prompt)
            print(answer)

def base_steering(llama: Llama, questions: pd.DataFrame):
    database = Database()
    timer = time.time()
    steered_data = database.get_steering_generation_of_llm(llama.model_id, "base")
    steered_data = steered_data[steered_data["question_id"].isin(questions["id"].values.tolist())]
    if len(steered_data) < len(questions):
        if len(steered_data) > 0:
            print(
                f"Base steering is incomplete ({len(steered_data)} < {len(questions)}); deleting data and recomputing"
            )
            database.remove_steering_generation_of_llm(llama.model_id, "base")
    else:
        print("Base steering already done; skipping")
        return
    system_prompt = prompts.get_base_steering_system_prompt()
    answers, llama_prompts = llama.batch_generate(
        questions["question"].tolist(),
        random_seed=42,
        system_prompt=system_prompt,
        return_prompts=True,
    )
    
    steered_data = pd.DataFrame(
        {
            "question_id": questions["id"],
            "steering_type": "base",
            "llm_name": llama.model_id,
            "answer": answers,
            "prompt": llama_prompts,
        }
    )
    log_time("Base steering", start_time=timer, logger=logger)
    database.insert_steering_generation(steered_data)
    print(database.get_steering_generation_of_llm(llama.model_id, "base")["answer"])


def group_steering(llama: Llama, questions: pd.DataFrame, group_names: List[str]):
    database = Database()
    timer = time.time()
    steered_data = database.get_steering_generation_of_llm(llama.model_id, "prompt_group")
    steered_data = steered_data[steered_data["question_id"].isin(questions["id"].values.tolist())]
    if len(steered_data) < len(questions) * len(group_names):
        if len(steered_data) > 0:
            print(
                f"Prompt group steering is incomplete ({len(steered_data)} < {len(questions)} * {len(group_names)}); deleting data and recomputing"
            )
            database.remove_steering_generation_of_llm(llama.model_id, "prompt_group")
    else:
        print("Prompt group steering already done; skipping")
        return
    steered_data = pd.DataFrame()
    answer_list = []
    prompt_list = []
    question_group_list = []
    for group_name in group_names:
        system_prompt = get_system_prompt_group(group_name)
        if system_prompt is None:
            logger.error(
                f"Skipping group steering for group {group_name} as it is not in GROUP_STEERING_NAMES"
            )
            continue
        
        question_group_list.append(
            ([group_name] * len(questions), questions["id"].tolist())
        )
        answers, llama_prompts = llama.batch_generate(
            questions["question"].tolist(),
            random_seed=42,
            system_prompt=system_prompt,
            return_prompts=True,
        )
        answer_list.append(answers)
        prompt_list.append(llama_prompts)

    answers = [answer for sublist in answer_list for answer in sublist]
    llama_prompts = [prompt for sublist in prompt_list for prompt in sublist]
    question_ids = [
        question_id for sublist in question_group_list for question_id in sublist[1]
    ]
    question_groups = [group for sublist in question_group_list for group in sublist[0]]
    steered_data = pd.DataFrame(
        {
            "question_id": question_ids,
            "data_group": question_groups,
            "steering_type": "prompt_group",
            "llm_name": llama.model_id,
            "answer": answers,
            "prompt": llama_prompts,
        }
    )
    log_time("Prompt group steering", start_time=timer, logger=logger)
    database.insert_steering_generation(steered_data)
    print(database.get_steering_generation_of_llm(llama.model_id)["answer"])

def get_attributes_per_group(group_names: List[str]):
    database = Database()   

    answers = database.get_answers(
        ["sfam_train", "sfam_val", "sfam_test", "lisa_train", "lisa_val", "lisa_test"]
    )
    
    answers = answers[answers[STYLE_EMBEDDING_VECTOR].notnull()]
    answers = answers[answers["data_group"].isin(group_names)]
    style_vector_df = database.get_style_vector()
    most_important_attribute_indeces = _get_most_important_attributes(
        np.array(answers[STYLE_EMBEDDING_VECTOR].values.tolist()),
        answers["data_group"].values,
    )
    groups = []
    style_attributes_list = []
    knowledge_attributes_list = []
    for group in group_names:
        # check if the group is in the steering names
        # not because it is needed, but too ensure that all steering is skipped if it is not included
        pretty_group_name = GROUP_STEERING_NAMES.get(group, None)
        if pretty_group_name is None:
            logger.error(
                f"Skipping attribute steering for group {group} as it is not in GROUP_STEERING_NAMES"
            )
            continue
        if group not in most_important_attribute_indeces:
            logger.error(
                f"There is no data to extract the most important attributes for the group {group}"
            )
            continue
        
        attribute_indeces = most_important_attribute_indeces[group]["dimensions"]
        attributes_correlations = most_important_attribute_indeces[group]["correlations"]
        
        
        
        # Get the most important attributes
        style_vector_data = style_vector_df.values
        group_style_vector = pd.DataFrame(
            style_vector_data[attribute_indeces],
            style_vector_df.index[attribute_indeces],
            style_vector_df.columns,
        )
        
        # Create a Series for correlations indexed by attribute indices
        correlation_series = pd.Series(
            data=attributes_correlations,
            index=style_vector_df.index[attribute_indeces],
            name="correlation"
        )

        # Add the correlation column to group_style_vector
        group_style_vector = group_style_vector.join(correlation_series)
        
        style_attributes_df = group_style_vector[group_style_vector["prompt_type"] == "Style prompt"]
        knowledge_attributes_df = group_style_vector[group_style_vector["prompt_type"] == "Knowledge prompt"]
        
        groups.append(group)
        style_attributes_list.append(style_attributes_df)
        knowledge_attributes_list.append(knowledge_attributes_df)
    return groups, style_attributes_list, knowledge_attributes_list
        
    

#def attribute_steering(llama: Llama, questions: pd.DataFrame, group_names: List[str]):
#    database = Database()
#    timer = time.time()
#    steered_data = database.get_steering_generation_of_llm(llama.model_id, "prompt_attribute")
#    steered_data = steered_data[steered_data["question_id"].isin(questions["id"].values.tolist())]
#    if len(steered_data) < len(questions) * len(group_names):
#        if len(steered_data) > 0:
#            print(
#                f"Prompt attribute steering is incomplete ({len(steered_data)} < {len(questions)} * {len(group_names)}); deleting data and recomputing"
#            )
#            database.remove_steering_generation_of_llm(llama.model_id, "prompt_attribute")
#    else:
#        print("Prompt attribute steering already done; skipping")
#        return
#
#    answers = database.get_answers(
#        ["sfam_train", "sfam_val", "sfam_test", "lisa_train", "lisa_val", "lisa_test"]
#    )
#    answers = answers[answers[STYLE_EMBEDDING_VECTOR].notnull()]
#    answers = answers[answers["data_group"].isin(group_names)]
#    style_vector_df = database.get_style_vector()
#    nlp = spacy.load("en_core_web_sm")
#    most_important_attribute_indeces = _get_most_important_attributes(
#        np.array(answers[STYLE_EMBEDDING_VECTOR].values.tolist()),
#        answers["data_group"].values,
#    )
#    answer_list = []
#    prompt_list = []
#    question_group_list = []
#    for group in group_names:
#        # check if the group is in the steering names
#        # not because it is needed, but too ensure that all steering is skipped if it is not included
#        pretty_group_name = GROUP_STEERING_NAMES.get(group, None)
#        if pretty_group_name is None:
#            logger.error(
#                f"Skipping attribute steering for group {group} as it is not in GROUP_STEERING_NAMES"
#            )
#            continue
#        if group not in most_important_attribute_indeces:
#            logger.error(
#                f"There is no data to extract the most important attributes for the group {group}"
#            )
#            continue
#        system_prompt = get_system_prompt_attribute(
#            nlp, style_vector_df, most_important_attribute_indeces[group]["dimensions"]
#        )
#        
#        question_group_list.append(([group] * len(questions), questions["id"].tolist()))
#        answers, llama_prompts = llama.batch_generate(
#            prompt_list,
#            random_seed=42,
#            system_prompt=system_prompt,
#            return_prompts=True,
#        )
#        answer_list.append(answers)
#        prompt_list.append(llama_prompts)
#        
#    answers = [answer for sublist in answer_list for answer in sublist]
#    llama_prompts = [prompt for sublist in prompt_list for prompt in sublist]
#    question_ids = [
#        question_id for sublist in question_group_list for question_id in sublist[1]
#    ]
#    question_groups = [group for sublist in question_group_list for group in sublist[0]]
#    steered_data = pd.DataFrame(
#        {
#            "question_id": question_ids,
#            "data_group": question_groups,
#            "steering_type": "prompt_attribute",
#            "llm_name": llama.model_id,
#            "answer": answers,
#            "prompt": llama_prompts,
#        }
#    )
#    log_time("Prompt attribute steering", start_time=timer, logger=logger)
#    database.insert_steering_generation(steered_data)


def group_attribute_steering(
    llama: Llama, questions: pd.DataFrame, group_names: List[str], num_important_attributes: int
):
    database = Database()
    timer = time.time()
    steered_data = database.get_steering_generation_of_llm(llama.model_id, f"prompt_group_attribute_{num_important_attributes}")
    steered_data = steered_data[steered_data["question_id"].isin(questions["id"].values.tolist())]
    if len(steered_data) < len(questions) * len(group_names):
        if len(steered_data) > 0:
            print(
                f"Prompt group attribute steering is incomplete ({len(steered_data)} < {len(questions)} * {len(group_names)}); deleting data and recomputing"
            )
            database.remove_steering_generation_of_llm(llama.model_id, f"prompt_group_attribute_{num_important_attributes}")
    else:
        print("Prompt group attribute steering already done; skipping")
        return

    answers = database.get_answers(
        ["sfam_train", "sfam_val", "sfam_test", "lisa_train", "lisa_val", "lisa_test"]
    )
    answers = answers[answers[STYLE_EMBEDDING_VECTOR].notnull()]
    style_vector_df = database.get_style_vector()
    nlp = spacy.load("en_core_web_sm")

    most_important_attribute_indeces = _get_most_important_attributes(
        np.array(answers[STYLE_EMBEDDING_VECTOR].values.tolist()),
        answers["data_group"].to_numpy(),
    )
    steered_data = pd.DataFrame()
    prompt_list = []
    answer_list = []
    question_group_list = []
    for group in group_names:
        if group not in most_important_attribute_indeces:
            logger.error(
                f"There is no data to extract the most important attributes for the group {group}"
            )
            continue
        system_prompt = get_system_prompt_group_attribute(
            num_important_attributes,
            nlp,
            group,
            style_vector_df,
            most_important_attribute_indeces[group]["dimensions"],
        )
        if system_prompt is None:
            logger.error(
                f"Skipping group attribute steering for group {group} as it is not in GROUP_STEERING_NAMES"
            )
            continue

        question_group_list.append(([group] * len(questions), questions["id"].tolist()))
        answers, llama_prompts = llama.batch_generate(
            questions["question"].tolist(),
            random_seed=42,
            system_prompt=system_prompt,
            return_prompts=True,
        )
        answer_list.append(answers)
        prompt_list.append(llama_prompts)
    answers = [answer for sublist in answer_list for answer in sublist]
    llama_prompts = [prompt for sublist in prompt_list for prompt in sublist]
    question_ids = [
        question_id for sublist in question_group_list for question_id in sublist[1]
    ]
    question_groups = [group for sublist in question_group_list for group in sublist[0]]
    steered_data = pd.DataFrame(
        {
            "question_id": question_ids,
            "data_group": question_groups,
            "steering_type": f"prompt_group_attribute_{num_important_attributes}",
            "llm_name": llama.model_id,
            "answer": answers,
            "prompt": llama_prompts,
        }
    )
    log_time("Prompt group attribute steering", start_time=timer, logger=logger)
    database.insert_steering_generation(steered_data)
    print(database.get_steering_generation_of_llm(llama.model_id)["answer"])


def _get_most_important_attributes(
    style_vectors: np.ndarray, groups: np.ndarray
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Compute the most important attributes for each group using point-biserial correlation.

    Args:
        style_vectors (np.ndarray): The style vectors of the samples.
        groups (np.ndarray): The group labels for each sample.
    Returns:
        Dict[str, Dict[str, np.ndarray]]: A dictionary where the keys are group names and the values are dictionaries
            containing the indices of the dimensions ordered by importance and their corresponding correlation coefficients.
    """
    unique_groups = np.unique(groups)
    most_important_dims = {}

    # Loop over each group
    for group in unique_groups:
        # Create a binary indicator: 1 if the sample belongs to the current group, 0 otherwise.
        binary_labels = (groups == group).astype(int)

        # Initialize an array to store correlation coefficients for each dimension
        corr_coeffs = np.zeros(style_vectors.shape[1])

        # Compute the point-biserial correlation for each dimension
        for i in range(style_vectors.shape[1]):
            corr, _ = pointbiserialr(style_vectors[:, i], binary_labels)
            corr_coeffs[i] = corr  # correlation for dimension i

        top_dims = np.argsort(corr_coeffs)[::-1]
        top_corrs = corr_coeffs[top_dims]

        most_important_dims[group] = {"dimensions": top_dims, "correlations": top_corrs}

    return most_important_dims


def _sort_by_attribute(
    num_important_attributes: int, style_vector_df: pd.DataFrame, attribute_indeces: np.array
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    style_vector_data = style_vector_df.values
    group_style_vector = pd.DataFrame(
        style_vector_data[attribute_indeces],
        style_vector_df.index[attribute_indeces],
        style_vector_df.columns,
    )
    style_attributes_df = group_style_vector[
        group_style_vector["prompt_type"] == "Style prompt"
    ].head(num_important_attributes)
    knowledge_attributes_df = group_style_vector[
        group_style_vector["prompt_type"] == "Knowledge prompt"
    ].head(num_important_attributes)
    return style_attributes_df, knowledge_attributes_df


def convert_attribute_sentence(
    nlp: spacy.language.Language,
    sentence: str,
    perspective: Literal["you", "you_neg", "they", "I", "I_neg"] = "you",
) -> str:
    if perspective not in ["you", "they", "you_neg", "I", "I_neg"]:
        raise ValueError("Perspective must be either 'you', 'you_neg', 'I', 'I_neg' or 'they'.")
    if perspective == "I":
        sentence = re.sub(r"\bThe author's\b", "My", sentence)
        sentence = re.sub(r"\bThe author is\b", "I am", sentence)
        sentence = re.sub(r"\bThe author\b", "I", sentence)
    elif perspective == "I_neg":
        if "The author's" in sentence:
            sentence = input(sentence)
        else:
            sentence = re.sub(r"\bThe author is\b", "I am NOT", sentence)
            sentence = re.sub(r"\bThe author\b", "I do NOT", sentence)
    elif perspective == "you":
        sentence = re.sub(r"\bThe author's\b", "Your", sentence)
        sentence = re.sub(r"\bThe author is\b", "You are", sentence)
        sentence = re.sub(r"\bThe author\b", "You", sentence)
    elif perspective == "you_neg":
        if "The author's" in sentence:
            sentence = input(sentence)
        else:
            sentence = re.sub(r"\bThe author is\b", "You are NOT", sentence)
            sentence = re.sub(r"\bThe author\b", "You do NOT", sentence)
    elif perspective == "they":
        sentence = re.sub(r"\bThe author's\b", "Their", sentence)
        sentence = re.sub(r"\bThe author is\b", "They are", sentence)
        sentence = re.sub(r"\bThe author\b", "They", sentence)

    # Process the sentence with spaCy
    doc = nlp(sentence)
    new_tokens = []

    for token in doc:
        # If the token is a verb and is in the third-person singular present (VBZ)
        if token.pos_ == "VERB" and token.tag_ == "VBZ":
            new_text = token.lemma_
        else:
            new_text = token.text

        # Append the token text along with its original trailing whitespace
        new_tokens.append(new_text + token.whitespace_)

    # Reconstruct the sentence from tokens.
    return "".join(new_tokens)


def get_system_prompt_group(group_name: str) -> Union[str, None]:
    pretty_group_name = GROUP_STEERING_NAMES.get(group_name, None)
    if pretty_group_name is None:
        return None
    return prompts.get_group_steering_system_prompt(pretty_group_name)


def get_system_prompt_attribute(
    nlp: spacy.language.Language,
    style_vector_df: pd.DataFrame,
    most_important_attribute_indeces: np.ndarray,
) -> str:
    style_attributes_df, knowledge_attributes_df = _sort_by_attribute(
        style_vector_df, most_important_attribute_indeces
   )
    # TODO: should the attributes be sorted by correlation?
    transformed_style_attributes = [
        convert_attribute_sentence(nlp, attribute, perspective="they")
        for attribute in style_attributes_df["style_sentence"].tolist()
    ]
    transformed_knowledge_attributes = [
        convert_attribute_sentence(nlp, attribute, perspective="they")
        for attribute in knowledge_attributes_df["style_sentence"].tolist()
    ]
    system_prompt = prompts.get_attribute_steering_system_prompt(
        transformed_style_attributes,
        transformed_knowledge_attributes,
    )
    return system_prompt


def get_system_prompt_group_attribute(
    num_important_attributes: int,
    nlp: spacy.language.Language,
    group: str,
    style_vector_df: pd.DataFrame,
    most_important_attribute_indeces: np.ndarray,
) -> Union[str, None]:
    pretty_group_name = GROUP_STEERING_NAMES.get(group, None)
    if pretty_group_name is None:
        return None
    style_attributes_df, knowledge_attributes_df = _sort_by_attribute(
        num_important_attributes, style_vector_df, most_important_attribute_indeces
    )
    
    print(f"Create prompt for {group} based on the following indices:")
    print("style_attribute_indeces", style_attributes_df)
    print("knowledge_attribute_indeces", knowledge_attributes_df)
    
    # TODO: should the attributes be sorted by correlation?
    transformed_style_attributes = [
        convert_attribute_sentence(nlp, attribute, perspective="they")
        for attribute in style_attributes_df["style_sentence"].tolist()
    ]
    transformed_knowledge_attributes = [
        convert_attribute_sentence(nlp, attribute, perspective="they")
        for attribute in knowledge_attributes_df["style_sentence"].tolist()
    ]
    system_prompt = prompts.get_group_attribute_steering_system_prompt(
        pretty_group_name,
        transformed_style_attributes,
        transformed_knowledge_attributes,
    )
    return system_prompt
