from io import TextIOWrapper
from pathlib import Path
import shutil
from typing import Callable, Generator, List, Tuple
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import logging
from sklearn.metrics.pairwise import cosine_similarity  # type: ignore
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score  # type: ignore
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer  # type: ignore
from datasets import Dataset  # type: ignore
import torch
from scipy.spatial.distance import cdist
from contextlib import contextmanager
import re
from collections import defaultdict

from config_new_prompts import (
    GROUP_STEERING_NAMES,
    RESULT_FILE_FORMAT,
    RESULT_FOLDER,
    SFAM_MODEL,
    MODEL_OUTPUT_FOLDER,
    STYLE_EMBEDDING_VECTOR,
)
from helper import AutoModelWithSigmoid
from scripts import sfam
from scripts.steering.prompt_steering import _get_most_important_attributes
from utils.database import Database
import string
from scripts.classifier import predict_fit_score, _prepare_data, GroupSpecificityClassifier
from safetensors.torch import load_file
from tqdm import tqdm

logger = logging.getLogger(__name__)

# the order must not be changed, but the metrics can be freely commented out/in and will be included in _compute_steering_evaluations accordingly
STEERING_METRICS = (
    "steering_effect_lisa",
    "direction_error_lisa",
    "possible_steering_effect_lisa",
    "possible_improvement_lisa",
    "possible_improvement_with_penalty_lisa",
    "steering_effect_normal",
    "direction_error_normal",
    "possible_steering_effect_normal",
    "semantic_alignment_normal",
    "semantic_alignment_with_penalty_normal",
    "group_score",
    "group_score_with_penalty"
    #"factual_accuracy",
)

STEERING_METRICS_TEX_NAMES = {
    "Possible Steering Effect": "Possible\\\\Steering Effect",
    "Steering Effect": "Steering Effect",
    "Direction Correctness": "Direction\\\\Correctness",
    "Possible Improvement": "Possible\\\\Improvement",
    "Distance Difference": "Distance\\\\Difference",
    "First 10 Attributes": "First 10\\\\Attributes",
    "First 20 Attributes": "First 20\\\\Attributes",
    "First 50 Attributes": "First 50\\\\Attributes",
}

def evaluate_models(do_evaluate_sfam=True):
    database = Database()
    file_name = f"results_models_{database.file_name}.tex" if RESULT_FILE_FORMAT == "latex" else f"results_models_{database.file_name}.md"
    #shutil.rmtree(RESULT_FOLDER, ignore_errors=True)
    Path.mkdir(Path(RESULT_FOLDER), exist_ok=True)
    with open(RESULT_FOLDER + file_name, "w") as file:
        #evaluate_knowledge_importance(file)
        evaluate_sfam(file, do_evaluate_sfam)
        evaluate_lisa(file)
        evaluate_embedder(file)
        evaluate_classifier(file)

def evaluate_all(steering_model):
    database = Database()
    file_name = f"results_{database.file_name}.tex" if RESULT_FILE_FORMAT == "latex" else f"results_{database.file_name}.md"
    #shutil.rmtree(RESULT_FOLDER, ignore_errors=True)
    Path.mkdir(Path(RESULT_FOLDER), exist_ok=True)
    with open(RESULT_FOLDER + file_name, "w") as file:
        evaluate_steering(file, steering_model)


def evaluate_knowledge_importance(file: TextIOWrapper):
    database = Database()
    logger.info("starting knowledge importance evaluation")
    answers = database.get_answers(
        ["sfam_test", "sfam_val", "sfam_train", "lisa_train", "lisa_val", "lisa_test"]
    )
    answers = answers[~answers[STYLE_EMBEDDING_VECTOR].isnull()]
    most_important_attribute_indeces = _get_most_important_attributes(
        np.array(answers[STYLE_EMBEDDING_VECTOR].values.tolist()),
        answers["data_group"].to_numpy(),
    )
    style_vector_df = database.get_style_vector()
    values = []
    for group in answers["data_group"].unique():
        attribute_ids = most_important_attribute_indeces[group]["dimensions"]
        group_values = []
        for num in [10, 20, 50]:
            num_knowledge_attributes = len(
                style_vector_df[
                    style_vector_df["style_vector_index"].isin(attribute_ids[:num])
                    & (style_vector_df["prompt_type"] == "Knowledge prompt")
                ]
            )
            group_values.append(num_knowledge_attributes)
        values.append([group] + group_values)
    df = pd.DataFrame(
        values,
        columns=[
            "Group",
            "First 10 Attributes",
            "First 20 Attributes",
            "First 50 Attributes",
        ],
    )
    with result_output(
        file, "Knowledge Importance Evaluation", "resultKnowledgeImportance"
    ) as write_dataframe:
        write_dataframe(df, full_width=True)


def evaluate_sfam(file: TextIOWrapper, do_evaluation=True):
    database = Database()
    with result_output(file, "SFAM evaluation", "resultSfam") as write_dataframe:
        if not do_evaluation:
            file.write("% Skipped evaluation\n\n")
            logger.warning("Skipping the evaluation of SFAM")
            return
        logger.info("starting sfam evaluation")
        tokenizer = AutoTokenizer.from_pretrained(SFAM_MODEL, use_fast=False)
        model = AutoModelWithSigmoid.from_pretrained(
            MODEL_OUTPUT_FOLDER + "sfam",
            dtype=torch.bfloat16
            #attn_implementation="flash_attention_2",
        )
        model.to("cuda" if torch.cuda.is_available() else "cpu")
        model.eval()

        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=sfam.INPUT_MAX_LENGTH,
            )

        test_data = Dataset.from_pandas(
            sfam._get_dataset_restricted_to_accepcted_clusters(data_type="sfam_test", max_num_samples=2)
        )
        test_data = test_data.map(tokenize_function, batched=True)
        test_data.set_format(
            type="torch", columns=["text", "input_ids", "attention_mask", "label"]
        )

        all_preds = []
        all_labels = []
        for batch in test_data:
            input_ids = (
                batch["input_ids"].unsqueeze(0).to("cuda")
            )  # Add batch dimension
            attention_mask = batch["attention_mask"].unsqueeze(0).to("cuda")
            label = batch["label"]

            with torch.no_grad():
                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                prediction = torch.round(logits).item()

            all_preds.append(prediction)
            all_labels.append(label)

        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)

        df = pd.DataFrame(
            [["Baseline Accuracy", 0.5], ["Accuracy", accuracy], ["F1", f1]]
        )
        write_dataframe(df, print_header=False, highlight_max=False)
        
        answers = database.get_answers(["sfam_train", "sfam_val", "sfam_test"])
        answers = answers[answers["style_vector_data"].notnull()]
        style_vector_sfam = np.array(answers["style_vector_sfam"].values.tolist())
        style_vector_data = np.array(answers["style_vector_data"].values.tolist())
        mse = np.mean(np.mean(np.square(style_vector_sfam - style_vector_data), axis=1))
        mae = np.mean(np.mean(np.abs(style_vector_sfam - style_vector_data), axis=1))
        cos_sim = np.mean(
            np.mean(cosine_similarity(style_vector_sfam, style_vector_data), axis=1)
        )
        style_vector_sfam = np.round(style_vector_sfam.flatten())
        style_vector_data = np.round(style_vector_data.flatten())
        accuracy = balanced_accuracy_score(style_vector_sfam, style_vector_data)
        f1 = f1_score(style_vector_sfam, style_vector_data, average='weighted')
        df = pd.DataFrame(
            [
                ["Mean Squared Error", mse],
                ["Mean Absolute Error", mae],
                ["Average Cosine Similarity", cos_sim],
                ["Baseline Accuracy", 0.5],
                ["Accuracy", accuracy],
                ["F1", f1],
            ]
        )
        with result_output(file, "SFAM evaluation", "resultSFAN") as write_dataframe:
            write_dataframe(df, highlight_max=False, print_header=False)


def evaluate_lisa(file: TextIOWrapper):
    database = Database()
    logger.info("starting lisa evaluation")
    answers = database.get_answers(["lisa_test"])
    answers = answers[answers["style_vector_sfam"].notnull()]
    style_vector_sfam = np.array(answers["style_vector_sfam"].values.tolist())
    style_vector_lisa = np.array(answers["style_vector_lisa"].values.tolist())
    mse = np.mean(np.mean(np.square(style_vector_sfam - style_vector_lisa), axis=1))
    mae = np.mean(np.mean(np.abs(style_vector_sfam - style_vector_lisa), axis=1))
    cos_sim = np.mean(
        np.mean(cosine_similarity(style_vector_sfam, style_vector_lisa), axis=1)
    )
    style_vector_sfam = np.round(style_vector_sfam.flatten())
    style_vector_lisa = np.round(style_vector_lisa.flatten())
    accuracy = balanced_accuracy_score(style_vector_sfam, style_vector_lisa)
    f1 = f1_score(style_vector_sfam, style_vector_lisa, average='weighted')
    df = pd.DataFrame(
        [
            ["Mean Squared Error", mse],
            ["Mean Absolute Error", mae],
            ["Average Cosine Similarity", cos_sim],
            ["Baseline Accuracy", 0.5],
            ["Accuracy", accuracy],
            ["F1", f1],
        ]
    )
    with result_output(file, "LISA evaluation", "resultLisa") as write_dataframe:
        write_dataframe(df, highlight_max=False, print_header=False)


def evaluate_embedder(file: TextIOWrapper):
    database = Database()
    logger.info("starting embedder evaluation")
    answers = database.get_answers(["lisa_test"])  # lisa test is embedding test as well
    answers = answers[~answers["style_embedding"].isnull()]
    df = _evaluate_embedder_like_training(answers)
    with result_output(
        file, "Embedder evaluation", "resultEmbedder"
    ) as write_dataframe:
        write_dataframe(df, print_header=False, highlight_max=False, full_width=True)

    training_answers = database.get_answers(
        [
            "sfam_train",
            "sfam_val",
            "sfam_test",
            "lisa_train",
            # "lisa_val",
        ]
    )
    groups = training_answers["data_group"].unique()
    groups = sorted(groups)
    style_vector_df = database.get_style_vector()
    mean_embeddings = np.empty(
        shape=(len(groups), len(style_vector_df)), dtype=np.float32
    )
    for i, group in enumerate(groups):
        group_answers = training_answers[training_answers["data_group"] == group]
        group_embedding = np.mean(
            np.array(group_answers["style_embedding"].values.tolist()), axis=0
        )
        mean_embeddings[i] = group_embedding

    df = _evaluate_embedder_to_mean(answers, mean_embeddings, groups)
    with result_output(
        file, "Embedder evaluation to group means", "resultEmbedderToMeans"
    ) as write_dataframe:
        write_dataframe(df, print_header=False, highlight_max=False, full_width=True)

    #######################################

    #answers = database.get_answers(["foreign_domain_test"])
    #answers = answers[~answers["style_embedding"].isnull()]
#
    #df = _evaluate_embedder_like_training(answers)
    #with result_output(
    #    file, "Embedder evaluation on foreign domain", "resultEmbedderForeignDomain"
    #) as write_dataframe:
    #    write_dataframe(df, print_header=False, highlight_max=False, full_width=True)
#
    #train_df, test_df = train_test_split(
    #    answers, test_size=0.5, stratify=answers["data_group"], random_state=42
    #)
    #groups = train_df["data_group"].unique()
    #groups = sorted(groups)
    #mean_embeddings = np.empty(
    #    shape=(len(groups), STYLE_EMBEDDING_SIZE), dtype=np.float32
    #)
    #for i, group in enumerate(groups):
    #    group_answers = train_df[train_df["data_group"] == group]
    #    group_embedding = np.mean(
    #        np.array(group_answers["style_embedding"].values.tolist()), axis=0
    #    )
    #    mean_embeddings[i] = group_embedding
#
    #df = _evaluate_embedder_to_mean(test_df, mean_embeddings, groups)
    #with result_output(
    #    file,
    #    "Embedder evaluation to group means on foreign domain",
    #    "resultEmbedderTomeansForeignDomain",
    #) as write_dataframe:
    #    write_dataframe(df, print_header=False, highlight_max=False, full_width=True)


def _evaluate_embedder_to_mean(
    answers: pd.DataFrame, mean_embeddings: np.ndarray, groups: List[str]
) -> pd.DataFrame:
    answer_length = len(answers)
    answers = answers[answers["data_group"].isin(groups)]
    if len(answers) != answer_length:
        logger.warning(
            f"{answer_length - len(answers)} of {answer_length} answers are not evaluated because they do not belong to known groups"
        )
    if answers.empty:
        logger.error("No answers to evaluate, skipping evaluation")
        return pd.DataFrame()
    answers = answers.sort_values("data_group")
    embeddings = np.array(answers["style_embedding"].values.tolist())
    embedding_group_indeces = np.array(
        [groups.index(group) for group in answers["data_group"].values.tolist()]
    )

    def correctness(embeddings, mean_embeddings, embedding_group_indeces, metric):
        distances = cdist(embeddings, mean_embeddings, metric=metric)
        closest_group = np.argmin(distances, axis=1)
        correct_group = closest_group == embedding_group_indeces
        return correct_group

    accuracy_euclidean = accuracy_score(
        correctness(
            embeddings, mean_embeddings, embedding_group_indeces, "euclidean"
        ),
        np.ones(len(answers), dtype=bool),
    )
    accuracy_cosine = accuracy_score(
        correctness(embeddings, mean_embeddings, embedding_group_indeces, "cosine"),
        np.ones(len(answers), dtype=bool),
    )
    f1_euclidean = f1_score(
        correctness(
            embeddings, mean_embeddings, embedding_group_indeces, "euclidean"
        ),
        np.ones(len(answers), dtype=bool),
    )
    f1_cosine = f1_score(
        correctness(embeddings, mean_embeddings, embedding_group_indeces, "cosine"),
        np.ones(len(answers), dtype=bool),
    )
    df = pd.DataFrame(
        [
            ["Baseline Accuracy", 1 / len(groups)],
            ["Accuracy (euclidean)", accuracy_euclidean],
            ["F1 (euclidean)", f1_euclidean],
            ["Accuracy (cosine)", accuracy_cosine],
            ["F1 (cosine)", f1_cosine],
        ]
    )
    return df


def _evaluate_embedder_like_training(answers: pd.DataFrame) -> pd.DataFrame:
    selected_answers = pd.DataFrame()
    not_selected_answers = pd.DataFrame()
    answer_groups = answers.groupby("data_group")
    for _, group_answers in answer_groups:
        num_anchors = len(group_answers) // 2
        group_selected = group_answers.sample(num_anchors)
        selected_answers = pd.concat([selected_answers, group_selected])
        not_selected_answers = pd.concat(
            [not_selected_answers, group_answers.drop(group_selected.index)]
        )
    result = []
    for group_name, group in selected_answers.groupby("data_group"):
        positive = not_selected_answers[
            not_selected_answers["data_group"] == group_name
        ].sample(len(group))
        negative = not_selected_answers[
            not_selected_answers["data_group"] != group_name
        ].sample(len(group))
        anchor_vectors = group["style_embedding"].values.tolist()
        positive_vectors = positive["style_embedding"].values.tolist()
        negative_vectors = negative["style_embedding"].values.tolist()
        result.extend(
            [
                [a, p, z]
                for (a, p, z) in zip(anchor_vectors, positive_vectors, negative_vectors)
            ]
        )

    anchor_embeddings = np.array([x[0] for x in result])
    positive_embeddings = np.array([x[1] for x in result])
    negative_embeddings = np.array([x[2] for x in result])

    pos_dist_euclidean = np.linalg.norm(anchor_embeddings - positive_embeddings, axis=1)
    neg_dist_euclidean = np.linalg.norm(anchor_embeddings - negative_embeddings, axis=1)
    pos_dist_cosine = (
        1 - cosine_similarity(anchor_embeddings, positive_embeddings).diagonal()
    )
    neg_dist_cosine = (
        1 - cosine_similarity(anchor_embeddings, negative_embeddings).diagonal()
    )

    accuracy_euclidean = accuracy_score(
        pos_dist_euclidean < neg_dist_euclidean, np.ones(len(result), dtype=bool)
    )
    accuracy_cosine = accuracy_score(
        pos_dist_cosine < neg_dist_cosine, np.ones(len(result), dtype=bool)
    )
    f1_euclidean = f1_score(
        pos_dist_euclidean < neg_dist_euclidean, np.ones(len(result), dtype=bool)
    )
    f1_cosine = f1_score(
        pos_dist_cosine < neg_dist_cosine, np.ones(len(result), dtype=bool)
    )
    df = pd.DataFrame(
        [
            ["Baseline Accuracy", 0.5],
            ["Accuracy (euclidean)", accuracy_euclidean],
            ["F1 (euclidean)", f1_euclidean],
            ["Accuracy (cosine)", accuracy_cosine],
            ["F1 (cosine)", f1_cosine],
        ]
    )
    return df

def evaluate_classifier(file: TextIOWrapper):
    model_path = f"{MODEL_OUTPUT_FOLDER}classifier"
    
    database = Database()
    style_vector_df = database.get_style_vector()
    groups = sorted(database.get_groups(
        ["sfam_train", "sfam_val", "sfam_test", "lisa_train", "lisa_val", "lisa_test"]
    ))
    model = GroupSpecificityClassifier(input_dim=len(style_vector_df), num_classes=len(groups))
    model.load_state_dict(load_file(f"{model_path}/model.safetensors"))
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    dataset = _prepare_data(groups)
    
    predictions = []
    fit_scores = []
    ground_truths = []

    for sample in dataset["test"]:
        attribute_vectors = sample["attribute_vectors"].to("cuda")
        labels = sample["labels"].to("cuda")

        with torch.no_grad():
            outputs = model(attribute_vectors)
            logits = outputs["logits"]  # raw logits
            
            # Convert logits to probabilities
            probs = torch.softmax(logits, dim=-1)
            
            # Predicted class
            pred_classes = probs.argmax(dim=-1)
            
            # Fit score of ground-truth class
            fit_score = probs[labels]

        predictions.append(pred_classes.cpu().numpy())
        fit_scores.append(fit_score.cpu().numpy())
        ground_truths.append(labels.cpu().numpy())
        
    mean_fit_score = np.mean(fit_scores)
    
    accuracy = accuracy_score(ground_truths, predictions)
    f1 = f1_score(ground_truths, predictions, average="macro")
    
    df = pd.DataFrame(
        [
            ["Baseline Accuracy", 1/len(groups)],
            ["Accuracy", accuracy],
            ["F1", f1],
            ["Fit score", mean_fit_score],
        ]
    )
    with result_output(file, "Classifier evaluation", "resultClassifier") as write_dataframe:
        write_dataframe(df, highlight_max=False, print_header=False)

def evaluate_steering(file: TextIOWrapper, steering_model):
    logger.info("starting steering evaluation")
    database = Database()
    steered_data = database.get_steering_generation_of_llm(llm_name=steering_model)
    old_length = len(steered_data)
    steered_data = steered_data[steered_data["normal_embedding"].notnull()]
    if len(steered_data) == 0:
        logger.info("No steered data found; skipping evaluate_steering")
        return
    if len(steered_data) != old_length:
        logger.warning(
            f"{old_length - len(steered_data)} steered generations without embeddings will not be evaluated in evaluate_steering"
        )

    # use only the training dataset to extract the group embeddings
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
    
    # Testing all steered data
    for test_data_split in ["eli5_test", "rooein_test"]:
        test_question_ids = database.get_steering_questions(split=test_data_split)["id"].unique()#[:20]
        test_steered_data = steered_data[
            steered_data["question_id"].isin(test_question_ids)
        ]
        test_steered_data = test_steered_data.sort_values("data_group")

        base_steering = test_steered_data[test_steered_data["steering_type"] == "base"]
        
        test_data_prompt = test_steered_data[
            test_steered_data["steering_type"].str.startswith("prompt_")
        ]
        test_data_prompt = test_data_prompt.sort_values("data_group")
        
        test_data_activation = test_steered_data[
            test_steered_data["steering_type"].str.startswith("activation_")
        ]
        test_data_activation = test_data_activation.sort_values("data_group")
        
        steered_data_complete = pd.concat(
            [
                test_data_prompt,
                test_data_activation,
            ]
        )
        steering_result, steering_types_order, target_group_order = _compute_steering_evaluations(
            steered_data_df=steered_data_complete,
            group_mean_embeddings_lisa=group_mean_embeddings_lisa,
            group_mean_embeddings_normal=group_mean_embeddings_normal,
            other_groups_mean_embeddings_normal=other_groups_mean_embeddings_normal,
            base_steering=base_steering,
        )
        
        np.save(RESULT_FOLDER + f"np_steering_results_{test_data_split}_{database.file_name}.npy", steering_result)
        
        # Check group order
        assert target_group_order_lisa == target_group_order
        assert target_group_order_normal == target_group_order
        
        # grouped by steering type
        df = pd.DataFrame(
            {
                "steering_type": steering_types_order,
                #"steering_effect_lisa": steering_result[STEERING_METRICS.index("steering_effect_lisa")].mean(axis=(1, 2)),
                "direction_error_lisa": steering_result[STEERING_METRICS.index("direction_error_lisa")].mean(axis=(1, 2)),
                #"possible_steering_effect_lisa": steering_result[STEERING_METRICS.index("possible_steering_effect_lisa")].mean(axis=(1, 2)),
                "possible_improvement_lisa": steering_result[STEERING_METRICS.index("possible_improvement_lisa")].mean(axis=(1, 2)),
                "possible_improvement_with_penalty_lisa": steering_result[STEERING_METRICS.index("possible_improvement_with_penalty_lisa")].mean(axis=(1, 2)),
                #"steering_effect_normal": steering_result[STEERING_METRICS.index("steering_effect_normal")].mean(axis=(1, 2)),
                "direction_error_normal": steering_result[STEERING_METRICS.index("direction_error_normal")].mean(axis=(1, 2)),
                #"possible_steering_effect_normal": steering_result[STEERING_METRICS.index("possible_steering_effect_normal")].mean(axis=(1, 2)),
                "semantic_alignment_normal": steering_result[STEERING_METRICS.index("semantic_alignment_normal")].mean(axis=(1, 2)),
                "semantic_alignment_with_penalty_normal": steering_result[STEERING_METRICS.index("semantic_alignment_with_penalty_normal")].mean(axis=(1, 2)),
                "group_score": steering_result[STEERING_METRICS.index("group_score")].mean(axis=(1, 2)),
                "group_score_with_penalty": steering_result[STEERING_METRICS.index("group_score_with_penalty")].mean(axis=(1, 2)),
            },
        )

        #question_id = test_question_ids[10]
        #group = "cs"
        #print()
        #print("Question:", steered_data_complete[steered_data_complete["question_id"] == question_id]["question"].values[0])
        #print("Target group:", "computer scientists")
        #print()
        ##print(steered_data_complete[(steered_data_complete["steering_type"] == "prompt_group_attribute_5") & (steered_data_complete["data_group"] == group)]["prompt"].values[0])
        #for index, steering_type in enumerate(sorted(steered_data_complete["steering_type"].unique())):
        #    print()
        #    print()
        #    #print(steering_type)
        #    print(f"Explanation {index+1}:")
        #    sample = steered_data_complete[
        #        (steered_data_complete["steering_type"] == steering_type) & 
        #        (steered_data_complete["question_id"] == question_id) &
        #        (steered_data_complete["data_group"] == group)
        #    ]["answer"].values
        #    print(sample[0])



        #df = df.sort_values(
        #    "steering_type",
        #    key=lambda x: x.map(
        #        {
        #            "prompt_group": 0,
        #            "prompt_attribute": 1,
        #            "prompt_group_attribute": 2,
        #            "activation_base": 3,
        #            "activation_group": 4,
        #            "activation_attribute": 5,
        #            "activation_group_attribute": 6,
        #        }
        #    ),
        #)
        #df = df.reset_index(drop=True)
        with result_output(
            file, f"Steering evaluation ({test_data_split})", "resultSteeringType"
        ) as write_dataframe:
            write_dataframe(df, full_width=True)
        
        try:
            # grouped by data group for each steering type
            for steering_type_index, steering_type in enumerate(steering_types_order):
                df = pd.DataFrame(
                    {
                        "group": target_group_order,
                        "direction_error_lisa": steering_result[STEERING_METRICS.index("direction_error_lisa"), steering_type_index, :, :].mean(axis=1),
                        "possible_improvement_lisa": steering_result[STEERING_METRICS.index("possible_improvement_lisa"), steering_type_index, :, :].mean(axis=1),
                        "possible_improvement_with_penalty_lisa": steering_result[STEERING_METRICS.index("possible_improvement_with_penalty_lisa"), steering_type_index, :, :].mean(axis=1),
                        "direction_error_normal": steering_result[STEERING_METRICS.index("direction_error_normal"), steering_type_index, :, :].mean(axis=1),
                        "semantic_alignment_normal": steering_result[STEERING_METRICS.index("semantic_alignment_normal"), steering_type_index, :, :].mean(axis=1),
                        "semantic_alignment_with_penalty_normal": steering_result[STEERING_METRICS.index("semantic_alignment_with_penalty_normal"), steering_type_index, :, :].mean(axis=1),
                        "group_score": steering_result[STEERING_METRICS.index("group_score"), steering_type_index, :, :].mean(axis=1),
                        "group_score_with_penalty": steering_result[STEERING_METRICS.index("group_score_with_penalty"), steering_type_index, :, :].mean(axis=1),
                    }
                )
                # for the presentation shorten the table
                # df = df.sort_values(
                #     ["cosine_similarity", "euclidean_distance"], ascending=[False, False]
                # )
                # df = pd.concat(
                #     [
                #         df.head(4),
                #         df.tail(4),
                #     ]
                # ).reset_index(drop=True)
                df = df.sort_values("group")
                with result_output(
                    file,
                    f"Steering evaluation (group, {steering_type})",
                    f"resultSteeringGroup{steering_type}",
                ) as write_dataframe:
                    write_dataframe(df, full_width=True)
        except Exception as e:
            print(e)

def _evaluate_activation_steering_joint_score(
    steering_type,
    steering_model,
    steered_data_df: pd.DataFrame,
    base_steering: pd.DataFrame,
    group_mean_embeddings_lisa: np.ndarray,
    group_mean_embeddings_normal: np.ndarray,
    other_groups_mean_embeddings_normal: np.ndarray
) -> Tuple[List[str], List[float]]:
    ACTIVATION_STEERING_METRICS = []
    # The activation-based steering validation results are only evaluated on group score (higher is better)
    activation_steering_metrics_string = ["group_score_with_penalty"]
    
    for metric in activation_steering_metrics_string:
        if "group_score" not in metric:
            print(f"WARNING: Skip {metric} since it is not a quality-based metrics = higher means not always better!")
            continue
        if metric in STEERING_METRICS:
            ACTIVATION_STEERING_METRICS.append(STEERING_METRICS.index(metric))
        else:
            exit("ERROR: At least one of the activation steering metrics is not in the overall list.")
    print("ACTIVATION_STEERING_METRICS INDEX", ACTIVATION_STEERING_METRICS)
    performance_array = np.empty(
        shape=(
            steered_data_df["activation_layer"].nunique(),
            steered_data_df["activation_lambda"].nunique(),
            len(ACTIVATION_STEERING_METRICS),
        )
    )
    joint_score_performance_array = np.empty(
        shape=(
            steered_data_df["activation_layer"].nunique(),
            steered_data_df["activation_lambda"].nunique(),
            len(ACTIVATION_STEERING_METRICS),
        )
    )
    layers = []
    lambdas = []
    steering_types_order = None
    target_group_order = None
    for layer_index, (_, layer_group) in enumerate(
        steered_data_df.groupby("activation_layer")
    ):
        layers.append(layer_group["activation_layer"].values[0])
        for lambda_index, (_, lambda_group) in enumerate(
            tqdm(layer_group.groupby("activation_lambda"), total=layer_group["activation_lambda"].nunique(), desc=f"Processing lambdas for layer {layers[layer_index]}:")
        ):
            #print(lambda_group)
            if layer_index == 0:
                lambdas.append(lambda_group["activation_lambda"].values[0])
            steering_performance, steering_types_order, target_group_order = _compute_steering_evaluations(
                steered_data_df=lambda_group,
                group_mean_embeddings_lisa=group_mean_embeddings_lisa,
                group_mean_embeddings_normal=group_mean_embeddings_normal,
                other_groups_mean_embeddings_normal=other_groups_mean_embeddings_normal,
                base_steering=base_steering,
            )
            
            # Calculate score based on all ACTIVATION_STEERING_METRICS (higher is better)
            steering_performance = steering_performance[
                ACTIVATION_STEERING_METRICS, :, :
            ]  
            # Caculate mean over all steering_types, data_groups and questions
            performance_array[layer_index, lambda_index, :] = np.mean(
                steering_performance, axis=(1, 2, 3)
            )
            
            # Calculate joint score based on mean of all ACTIVATION_STEERING_METRICS to get best hyperparameters (higher is better)
            steering_joint_score_performance = np.expand_dims(np.mean(steering_performance, axis=0), axis=0) 
            
            # Caculate mean over all steering_types, data_groups and questions
            joint_score_performance_array[layer_index, lambda_index, :] = np.mean(
                steering_joint_score_performance, axis=(1, 2, 3)
            )
            
    #print(layers, lambdas) 
    
    plot_activation_steering_performance(
        steering_model=steering_model,
        ACTIVATION_STEERING_METRICS=ACTIVATION_STEERING_METRICS,
        steering_type=steering_type,
        data=np.transpose(performance_array, (2, 0, 1)),
        values_1=layers,
        values_2=lambdas,
    )
    
    best_args = []
    for i in range(joint_score_performance_array.shape[2]):
        best_args.append(np.argmax(joint_score_performance_array[:, :, i]))
    # best_args is an np.ndarray of shape (4,) and contains the indices of the best performance for each metric
    best_args = np.unravel_index(best_args, joint_score_performance_array.shape[:2])
    return performance_array, layers, layers[best_args[0][0]], lambdas, lambdas[best_args[1][0]], steering_types_order, target_group_order

def is_ascii(text):
    replacements = {
        '\u2019': "'",  # right single quotation mark
        '\u2018': "'",  # left single quotation mark
        '\u201C': '"',  # left double quotation mark
        '\u201D': '"',  # right double quotation mark
        '\u2013': '-',  # en dash
        '\u2014': '-',  # em dash
        '\u2026': '...',  # ellipsis
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    try:
        text.encode('latin-1')
        return True
    except UnicodeEncodeError:
        return False

def _has_repeat(seq, min_unit=1, max_unit=50):
    """
    Detect consecutive repeated segments inside seq.
    seq may be a list of words or list of characters.
    min_unit: minimum segment size
    max_unit: maximum segment size (prevents O(n^2) blowup)
    """
    n = len(seq)
    max_unit = min(max_unit, n // 2)
    
    for size in range(min_unit, max_unit + 1):
        i = 0
        while i + 2 * size <= n:
            if seq[i:i+size] == seq[i+size:i+2*size]:
                return True
            i += 1
    return False


def _find_repetitive_patterns_word_chars(text):
    tokens = text.split()
    words = [t for t in tokens if t not in string.punctuation]
    # 1. Word-level
    if _has_repeat(words, min_unit=2):
        return True

    # 2. Character-level inside words 
    for w in words:
        chars = list(w)
        if len(chars) > 5 and _has_repeat(chars, min_unit=3, max_unit=len(chars)//2):
            return True

    return False

def find_repetitive_patterns(texts):
    repetition_penalty = []
    for text in texts:
        if not is_ascii(text):
            repetition_penalty.append(1)
            continue
        if _find_repetitive_patterns_word_chars(text):
            repetition_penalty.append(1)
            continue
        repetition_penalty.append(0)
    return np.array(repetition_penalty)

def get_length_penalty(texts, goal=200):
    length_penalty = []
    for text in texts:
        actual = len(text.split())
        mae = abs(actual - goal)
        if mae > 100:
            length_penalty.append(1)
        else:
            length_penalty.append(0)
    return np.array(length_penalty)

#def get_length_penalty(texts, min_length=100):
#    length_penalty = []
#    for text in texts:
#        actual_length = len(text.split())
#        if actual_length < min_length:
#            length_penalty.append(1)
#        else:
#            length_penalty.append(0)
#    return np.array(length_penalty)


def _compute_steering_evaluations(
    steered_data_df: pd.DataFrame,
    group_mean_embeddings_lisa: np.ndarray,
    group_mean_embeddings_normal: np.ndarray,
    other_groups_mean_embeddings_normal: np.ndarray,
    base_steering: pd.DataFrame,
) -> np.ndarray:
    # result is of shape (4, x, y, z) where x is the number of steering_types, y is the number of data_groups, and z is the number of questions
    # the first array is steering_effect, the second is steering_direction_correctness, the third is possible_steering_effect, and the fourth is possible_improvements
    result_shape = (
        len(STEERING_METRICS),
        steered_data_df["steering_type"].nunique(),
        steered_data_df["data_group"].nunique(),
        steered_data_df["question"].nunique(),
    )
    result = np.empty(shape=result_shape, dtype=np.float32)
    steering_types_order = []
    target_group_order = []
    for steering_type_index, (_, group) in enumerate(
        steered_data_df.groupby("steering_type")
    ):
        steering_types_order.append(group["steering_type"].values[0])
        for question_index, (question, question_group) in enumerate(
            group.groupby("question")
        ):
            if steering_type_index == 0 and question_index == 0:
                target_group_order = question_group["data_group"].values.tolist()
                #print(f"steering_types_order: {steering_types_order}")
                #print(f"target_group_order: {target_group_order}")
            # Penalties
            repetition_penalty = find_repetitive_patterns(question_group["answer"].values.tolist())
            length_penalty = get_length_penalty(question_group["answer"].values.tolist())
            
            ## Group score based on classifier (higher is better)
            fit_scores = predict_fit_score(question_group)
            result[STEERING_METRICS.index("group_score"), steering_type_index, :, question_index] = (
                    fit_scores
                )
            result[STEERING_METRICS.index("group_score_with_penalty"), steering_type_index, :, question_index] = (
                    fit_scores - 2*repetition_penalty - length_penalty
                )
            
            ## Results based on LISA embedder
            base_embedding_lisa = base_steering[base_steering["question"] == question][
                "style_embedding"
            ].values[0]
            #print(question_group["data_group"])
            steered_embeddings_lisa = np.array(
                question_group["style_embedding"].values.tolist()
            )
            
            steering_effect_vector_lisa = steered_embeddings_lisa - base_embedding_lisa
            possible_steering_effect_vector_lisa = group_mean_embeddings_lisa - base_embedding_lisa
            
            # Steering effect
            result[STEERING_METRICS.index("steering_effect_lisa"), steering_type_index, :, question_index] = np.linalg.norm(
                        steering_effect_vector_lisa, axis=1
                    )
            # Steering direction error
            result[STEERING_METRICS.index("direction_error_lisa"), steering_type_index, :, question_index] = (
                    (1 - cosine_similarity(
                        steering_effect_vector_lisa, possible_steering_effect_vector_lisa
                    ).diagonal())/2
                )
            # Possible steering effect
            result[STEERING_METRICS.index("possible_steering_effect_lisa"), steering_type_index, :, question_index] = np.linalg.norm(
                possible_steering_effect_vector_lisa, axis=1
            )
            # Possible improvement
            result[STEERING_METRICS.index("possible_improvement_lisa"), steering_type_index, :, question_index] = np.linalg.norm(
                group_mean_embeddings_lisa - steered_embeddings_lisa, axis=1
            )
            # Possible improvement with repetition and length penalty (lower is better)
            possible_improvement = np.linalg.norm(
                                            group_mean_embeddings_lisa - steered_embeddings_lisa, axis=1
                                        )
            result[STEERING_METRICS.index("possible_improvement_with_penalty_lisa"), steering_type_index, :, question_index] = (
                    possible_improvement + 10*repetition_penalty + 5*length_penalty
                )
            
            
            ## Results based on normal embedder
            base_embedding_normal = base_steering[base_steering["question"] == question][
                "normal_embedding"
            ].values[0]
            #print(question_group["data_group"])
            steered_embeddings_normal = np.array(
                question_group["normal_embedding"].values.tolist()
            )
            #print(group_mean_embeddings_normal.shape)
            #print(steered_embeddings_normal.shape)
            
            steering_effect_vector_normal = steered_embeddings_normal - base_embedding_normal
            possible_steering_effect_vector_normal = group_mean_embeddings_normal - base_embedding_normal
            
            # Steering effect
            result[STEERING_METRICS.index("steering_effect_normal"), steering_type_index, :, question_index] = (
                    (1 - cosine_similarity(
                        steered_embeddings_normal, base_embedding_normal.reshape(1, -1)
                    ).diagonal())/2
                )
            # Steering direction error
            result[STEERING_METRICS.index("direction_error_normal"), steering_type_index, :, question_index] = (
                    (1 - cosine_similarity(
                        steering_effect_vector_normal, possible_steering_effect_vector_normal
                    ).diagonal())/2
                )
            # Possible steering effect
            result[STEERING_METRICS.index("possible_steering_effect_normal"), steering_type_index, :, question_index] = (
                    (1 - cosine_similarity(
                        group_mean_embeddings_normal, base_embedding_normal.reshape(1, -1)
                    ).diagonal())/2
                )
             # Possible improvement (higher is better)
            result[STEERING_METRICS.index("semantic_alignment_normal"), steering_type_index, :, question_index] = (
                    cosine_similarity(
                        group_mean_embeddings_normal, steered_embeddings_normal
                    ).diagonal()
                )
            # Possible improvement with repetition and length penalty (higher is better)
            sim = cosine_similarity(
                            group_mean_embeddings_normal, steered_embeddings_normal
                        ).diagonal()
            
            result[STEERING_METRICS.index("semantic_alignment_with_penalty_normal"), steering_type_index, :, question_index]  = (
                    sim - 2*repetition_penalty - length_penalty
                )
            
            
            ## Results based on fact checker
            # TODO
            
    return result, steering_types_order, target_group_order

def plot_activation_steering_performance(
    steering_model, ACTIVATION_STEERING_METRICS: list, steering_type: str, data: np.ndarray, values_1: list, values_2: list
):
    database = Database()
    """
    Plot the activation steering performance.

    Args:
        data (np.ndarray): array of shape (`len(ACTIVATION_STEERING_METRICS)`, x, y) where x is the number of activation layers and y is the number of activation lambdas.
        The first array is steering_effect, the second is steering_direction_correctness, the third is possible_steering_effect, and the fourth is possible_improvements.
        values_1 (list): of activation layers or other values for the y-axis.
        The length of this list should match the first dimension of the data array.
        values_2 (list): of activation lambdas or other values for the x-axis.
        The length of this list should match the second dimension of the data array.
    """

    # Compute sorted order indices
    idx1 = sorted(range(len(values_1)), key=lambda i: [list(map(int, s.split(','))) for s in values_1][i])
    idx2 = np.argsort(values_2)
    # Apply sorted lists
    values_1 = [str(values_1[i]) for i in idx1]
    values_2 = [values_2[i] for i in idx2]

    print(values_1)
    print(values_2)
    
    # Reorder the data to match sorted labels
    data = data[:, idx1, :][:, :, idx2]
    
    for i in range(len(ACTIVATION_STEERING_METRICS)):
        plt.figure(figsize=(6, 5))
        plt.imshow(
            data[i],
            aspect="auto",
            origin="lower",
            cmap="viridis",
        )
        # plt.title(
        #     STEERING_METRICS[ACTIVATION_STEERING_METRICS[i]].replace("_", " ").title()
        # )
        plt.ylabel("Layer")
        plt.xlabel("Lambda")
        plt.xticks(np.arange(len(values_2)), values_2)
        plt.yticks(np.arange(len(values_1)), values_1)
        plt.colorbar()

        plt.tight_layout()
        if RESULT_FILE_FORMAT == "markdown":
            plt.savefig(
                RESULT_FOLDER
                + f"{database.file_name}_{steering_type}_{STEERING_METRICS[ACTIVATION_STEERING_METRICS[i]]}.png"
            )
        else:
            try:
                plt.rcParams.update(
                    {
                        "pgf.texsystem": "pdflatex",
                        "text.usetex": True,
                        "pgf.preamble": r"\usepackage{siunitx}",
                    }
                )
                plt.savefig(
                    RESULT_FOLDER
                    + f"{database.file_name}_{steering_type}_{STEERING_METRICS[ACTIVATION_STEERING_METRICS[i]]}.pgf"
                )
            except Exception as e:
                logger.error(
                    "Failed to save the activation steering performance plot as PGF, falling back to PNG format.",
                    exc_info=e,
                )
                plt.savefig(
                    RESULT_FOLDER
                    + f"{database.file_name}_{steering_type}_{STEERING_METRICS[ACTIVATION_STEERING_METRICS[i]]}.png"
                )


#def _compute_steering_evaluations(
#    steered_data_df: pd.DataFrame,
#    group_embeddings: np.ndarray,
#    base_steering: pd.DataFrame,
#) -> np.ndarray:
#    # result is of shape (4, x, y, z) where x is the number of steering_types, y is the number of data_groups, and z is the number of questions
#    # the first array is steering_effect, the second is steering_direction_correctness, the third is possible_steering_effect, and the fourth is possible_improvements
#    result_shape = (
#        len(STEERING_METRICS),
#        steered_data_df["steering_type"].nunique(),
#        steered_data_df["data_group"].nunique(),
#        steered_data_df["question"].nunique(),
#    )
#    result = np.empty(shape=result_shape, dtype=np.float32)
#    for steering_type_index, (_, group) in enumerate(
#        steered_data_df.groupby("steering_type")
#    ):
#        for question_index, (question, question_group) in enumerate(
#            group.groupby("question")
#        ):
#            base_embedding = base_steering[base_steering["question"] == question][
#                "style_embedding"
#            ].values[0]
#            steered_embeddings = np.array(
#                question_group["style_embedding"].values.tolist()
#            )
#            steering_effect_vector = steered_embeddings - base_embedding
#            possible_steering_effect_vector = group_embeddings - base_embedding
#            if len(STEERING_METRICS) > 0:
#                # steering effect
#                result[0, steering_type_index, :, question_index] = np.linalg.norm(
#                    steering_effect_vector, axis=1
#                )
#            if len(STEERING_METRICS) > 1:
#                # steering direction correctness
#                result[1, steering_type_index, :, question_index] = (
#                    cosine_similarity(
#                        steering_effect_vector, possible_steering_effect_vector
#                    ).diagonal()
#                    # cosine_similarity(steering_effect_vector, group_embeddings).diagonal()
#                )
#            if len(STEERING_METRICS) > 2:
#                # possible steering effect
#                result[2, steering_type_index, :, question_index] = np.linalg.norm(
#                    possible_steering_effect_vector, axis=1
#                )
#            if len(STEERING_METRICS) > 3:
#                # possible improvement
#                result[3, steering_type_index, :, question_index] = np.linalg.norm(
#                    group_embeddings - steered_embeddings, axis=1
#                )
#            if len(STEERING_METRICS) > 4:
#                # distance differences
#                result[4, steering_type_index, :, question_index] = np.linalg.norm(
#                    possible_steering_effect_vector, axis=1
#                ) - np.linalg.norm(steering_effect_vector, axis=1)
#    return result


def _df_to_latex(
    df: pd.DataFrame, highilight_max=True, print_header=True, full_width=False
) -> str:
    def _format_cell(x: object):
        if not isinstance(x, str):
            return x
        if x in GROUP_STEERING_NAMES.keys():
            # x = f"{GROUP_STEERING_NAMES[x].title()} ({x})"
            x = GROUP_STEERING_NAMES[x].title()
        else:
            x = x.title()
        x = x.replace(r"\_", " ")
        x = x.replace("_", " ")
        return x

    style_df = df.style.hide(axis=0)
    if not print_header:
        style_df = style_df.hide(axis=1)

    latex_string = (
        style_df.format(_format_cell, escape="latex")
        .format_index(
            _format_cell,
            axis=1,
            escape="latex",
        )
        .to_latex(siunitx=True, hrules=True)
    )
    if not print_header:
        latex_string = latex_string.replace("\\midrule\n", "")
    for name in STEERING_METRICS_TEX_NAMES.keys():
        latex_string = latex_string.replace(name, STEERING_METRICS_TEX_NAMES[name])

    cell_specs = ""
    if highilight_max:
        max_indices = defaultdict(list)
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                if col == "possible_improvement":
                    # Highlight the minimum value for possible improvement
                    min_value = df[col].min()
                    max_indices[col].extend(df.index[df[col] == min_value].tolist())
                else:
                    max_value = df[col].max()
                    max_indices[col].extend(df.index[df[col] == max_value].tolist())
        for col, indices in max_indices.items():
            col_idx = df.columns.get_loc(col) + 1
            for idx in indices:
                row_idx = idx + 1
                if print_header:
                    row_idx += 1  # Adjust for header row
                cell_specs += f"cell{{{row_idx}}}{{{col_idx}}} = gray9,\n"

    new_colspec_parts = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            new_colspec_parts.append("S")
        elif full_width:
            new_colspec_parts.append("X")
        else:
            new_colspec_parts.append("l")
    new_colspec = " ".join(new_colspec_parts)

    # Replace tabular environment with tblr and insert new colspec
    latex_string = re.sub(
        r"\\begin{tabular}{.*?}",
        rf"\\begin{{tblr}}{{\ncolspec={{ {new_colspec} }},\n{cell_specs}}}",
        latex_string,
    )
    latex_string = latex_string.replace("\\end{tabular}", "\\end{tblr}")

    return latex_string


@contextmanager
def result_output(
    file: TextIOWrapper, title: str, command_name: str
) -> Generator[Callable[[pd.DataFrame, bool, bool, bool], None], None, None]:
    def _write_dataframe(
        df: pd.DataFrame, highlight_max=True, print_header=True, full_width=False
    ):
        if RESULT_FILE_FORMAT == "latex":
            file.write(
                _df_to_latex(
                    df,
                    highilight_max=highlight_max,
                    print_header=print_header,
                    full_width=full_width,
                )
            )
        elif RESULT_FILE_FORMAT == "markdown":
            file.write(df.to_markdown(index=False))

    try:
        if RESULT_FILE_FORMAT == "latex":
            file.write(f"%%%%%% {title} %%%%%%" + "\n\n")
            file.write("\\newcommand{\\" + command_name + "}{" + "\n")
        elif RESULT_FILE_FORMAT == "markdown":
            file.write(f"## {title} \n\n")
        yield _write_dataframe
    finally:
        if RESULT_FILE_FORMAT == "latex":
            file.write("}")
        file.write("\n\n")


def plot_attribute_distribution():
    # TODO: use this function?
    database = Database()
    answer_with_vector_ids = pd.read_sql_query(
        """
        WITH attributes_per_answer AS (
            SELECT answers.id AS answer_id, COUNT(DISTINCT clusters.id) AS attribute_count
            FROM answers
            JOIN style_descriptions ON style_descriptions.answer_id = answers.id
            JOIN join_sentence_description ON join_sentence_description.description_id = style_descriptions.id
            JOIN style_sentences ON style_sentences.id = join_sentence_description.sentence_id
            JOIN clusters ON clusters.id = style_sentences.cluster_id
            WHERE clusters.status = 10
            AND answers.data_type IN ('sfam_train', 'sfam_val', 'sfam_test')
            GROUP BY answers.id
        )
        SELECT COUNT(answer_id) AS answer_count, attribute_count
        FROM attributes_per_answer
        GROUP BY attribute_count
        """,
        database._conn,
    )
    answer_with_vector_ids

    # create a distribution plot of the number of attributes per answer
    plt.figure(figsize=(6, 5))
    plt.bar(
        answer_with_vector_ids["attribute_count"],
        answer_with_vector_ids["answer_count"],
        color="blue",
        alpha=0.7,
    )
    plt.xlabel("Number of attribute vector dimensions used per answer")
    plt.ylabel("Number of answers")
    plt.title("Distribution of number of attributes per answer")
    # ticks every 5 from 0 to max attribute count
    plt.xticks(
        range(0, answer_with_vector_ids["attribute_count"].max() + 1, 5),
    )
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    # plt.savefig("number_of_attributes_per_answer.png", dpi=300)
    # old_backend = matplotlib.get_backend()
    # matplotlib.use("pgf")
    # plt.rcParams.update(
    #     {
    #         "pgf.texsystem": "pdflatex",
    #         "text.usetex": True,
    #         "pgf.preamble": r"\usepackage{siunitx}",
    #     }
    # )
    # plt.savefig(
    #     RESULT_FOLDER + "number_of_attributes_per_answer.pgf",
    #     bbox_inches="tight",
    #     pad_inches=0.1,
    # )
    # matplotlib.use(old_backend)
    try:
        plt.rcParams.update(
            {
                "pgf.texsystem": "pdflatex",
                "text.usetex": True,
                "pgf.preamble": r"\usepackage{siunitx}",
            }
        )
        plt.savefig(
            RESULT_FOLDER + "number_of_attributes_per_answer.pgf",
            bbox_inches="tight",
            pad_inches=0.1,
        )
    except Exception as e:
        logger.error(
            "Failed to save the attribute distribution plot as PGF, falling back to PNG format.",
            exc_info=e,
        )
        plt.savefig(
            RESULT_FOLDER + "number_of_attributes_per_answer.png",
            bbox_inches="tight",
            pad_inches=0.1,
        )
