import shutil
import time
from typing import Callable, Optional
import numpy as np
import pandas as pd
import torch
import logging
import os
from datasets import Dataset, DatasetDict
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    EvalPrediction,
    TrainerCallback
)
from sklearn.metrics import accuracy_score, f1_score

from config_new_prompts import (
    MODEL_OUTPUT_FOLDER,
    SFAM_MODEL,
    HPO_FILE,
    SFAM_MIN_ACCURACY,
)
from helper import (
    DATA_TYPE_ENUM,
    HyperparameterSearchLogger,
    log_time,
    AutoModelWithSigmoid,
)
from utils.database import Database
from utils.sentence_transformer import SentenceTransformerModel
from typing import Literal
from torch.utils.data import Sampler

import random

MODEL_OUTPUT_FOLDER = "models_qwen_old_prompt_new_training/"

logger = logging.getLogger(__name__)

INPUT_MAX_LENGTH = 512
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def train_and_create_style_vector(
    recompute=False, do_hyperparameter_optimization=False
):
    if recompute:
        delete_model()

    _train(do_hyperparameter_optimization=do_hyperparameter_optimization)
    _create_style_vector(f"{MODEL_OUTPUT_FOLDER}sfam")
    
def train(
    recompute=False, do_hyperparameter_optimization=False
):
    if recompute:
        delete_model()

    _train(do_hyperparameter_optimization=do_hyperparameter_optimization)
    
def predict_style_vector(batch):
    _create_style_vector(f"{MODEL_OUTPUT_FOLDER}sfam", batch)
    

def _create_style_vector(model_name: str, batch: Literal[0, 1, 2, 3, 4, 5]):
    database = Database()
    answer_count = database.count_answers_without_stylevector(
        vector_type="sfam",
        data_types=[
            "sfam_train",
            "sfam_val",
            "sfam_test",
            "lisa_train",
            "lisa_val",
            "lisa_test",
        ],
    )
    if answer_count == 0:
        print(
            "All answers have SFAM style vectors; skipping style vector generation"
        )
        return
    answers = database.get_answers(
        ["sfam_train", "sfam_val", "sfam_test", "lisa_train", "lisa_val", "lisa_test"]
    )
    print(len(answers))
    
    #mid = len(answers) // 6
    #answers = answers[batch*mid:(batch+1)*mid]
    #print(len(answers))
    answers = answers[answers["style_vector_sfam"].isnull()]
    print(len(answers))
    answers = answers[answers["style_vector_data"].notnull()]
    print(len(answers))
    
    create_style_vector(
        model_name=model_name,
        data=answers,
        save_callback=lambda result: database.update_answer_styleVector(
            result, "sfam", skip_logging=True
        ),
    )
    database.backup("sfam_style_vector")


def create_style_vector(
    model_name: str, data: pd.DataFrame, save_callback: Callable
) -> pd.DataFrame:
    SFAM_BATCH_SIZE = 1500
    database = Database()
    timer = time.time()
    print("Started creating SFAM style vector")
    style_vector_df = database.get_style_vector()
    tokenizer = AutoTokenizer.from_pretrained(SFAM_MODEL, use_fast=False)
    model = AutoModelWithSigmoid.from_pretrained(
        model_name,
        dtype=torch.bfloat16
        #attn_implementation="flash_attention_2",
    )
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    for _, answer_row in tqdm(
        data.iterrows(),
        total=len(data),
        desc="Create SFAM style vector for all answers",
        smoothing=0,
    ):
        input_texts = (
            style_vector_df["style_sentence"]
            .apply(lambda x: _format_sfam_input(x, answer_row["answer"]))
            .tolist()
        )
        style_vectors = np.array([])
        for index in range(0, len(input_texts), SFAM_BATCH_SIZE):
            batch = input_texts[index : index + SFAM_BATCH_SIZE]
            tokenized_batch = tokenizer(
                batch,
                padding="max_length",
                truncation=True,
                max_length=INPUT_MAX_LENGTH,
                return_tensors="pt",
            )
            tokenized_batch = {
                k: v.to("cuda" if torch.cuda.is_available() else "cpu")
                for k, v in tokenized_batch.items()
            }
            with torch.no_grad():
                outputs = model(**tokenized_batch)
            logits = outputs.logits
            agreement_scores = logits.squeeze().to(torch.float32).cpu().numpy()
            style_vectors = np.concatenate((style_vectors, agreement_scores))
        result_answers = pd.DataFrame(
            [
                {
                    "answer_id": answer_row["id"],
                    "style_vector": style_vectors.astype(np.float32),
                }
            ]
        )
        save_callback(result_answers)

    log_time(message="Creating SFAM style vector", start_time=timer, logger=logger)
    

def _train(do_hyperparameter_optimization) -> tuple[float, float]:
    print(MODEL_OUTPUT_FOLDER)
    if os.path.exists(f"{MODEL_OUTPUT_FOLDER}sfam"):
        print("SFAM model already exists, skipping training.")
        return -1.0, -1.0
    if not os.path.exists("sfam_data"):
        dataset = _prepare_data_sentences({"sfam_train": 2, "sfam_val": 2, "sfam_test": 2})
        dataset.save_to_disk("sfam_data")
    else:
        dataset = DatasetDict.load_from_disk("sfam_data")

    timer = time.time()
    print("Started training SFAM")
    # Load the tokenizer and model
    
    # compute loss function that provides the accuracy
    def compute_metrics(pred: EvalPrediction):
        logits: np.ndarray | tuple[np.ndarray] = pred.predictions
        predictions = np.round(logits)
        
        accuracy = accuracy_score(pred.label_ids, predictions)
        print(f"SFAM validation accuracy: {accuracy}", flush=True)
        return {
            "accuracy": accuracy,
            # "f1": f1_score(pred.label_ids, ),
        }

    # Define TrainingArguments
    training_args = TrainingArguments(
        output_dir=f"{MODEL_OUTPUT_FOLDER}checkpoints/sfam",  # output directory for model predictions and checkpoints
        save_total_limit=2,
        eval_strategy="steps",
        save_strategy="steps",
        logging_strategy="steps",
        num_train_epochs=3,  # number of training epochs
        logging_steps=500, 
        save_steps=500,
        learning_rate=3e-5,
        weight_decay=0.25,
        per_device_train_batch_size=12,  # batch size per device during training
        per_device_eval_batch_size=300,  # batch size per device during evaluation
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        dataloader_num_workers=8,
        dataloader_pin_memory=True,
        
    )
    
    class BestEvalLossCallback(TrainerCallback):
        def __init__(self):
            self.best_eval_loss = float("inf")

        def on_train_begin(self, args, state, control, **kwargs):
            # New trial starting
            self.best_eval_loss = float("inf")
            print("Set best_eval_loss to float('inf')!")

        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            if metrics is not None and "eval_loss" in metrics:
                current_loss = metrics["eval_loss"]

                if current_loss < self.best_eval_loss:
                    self.best_eval_loss = current_loss
                print(f"Best val loss so far: {self.best_eval_loss}")
            return control

    best_loss_callback = BestEvalLossCallback()

    # Initialize Trainer
    trainer = Trainer(
        model_init=lambda: AutoModelWithSigmoid.from_pretrained(
            SFAM_MODEL,
            num_labels=1,
            problem_type="single_label_classification",
            dtype=torch.bfloat16
        ).to("cuda" if torch.cuda.is_available() else "cpu"),
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=5),
            HyperparameterSearchLogger(logger),
            best_loss_callback
        ],
        compute_metrics=compute_metrics,
    )
    
    if do_hyperparameter_optimization:

        def hp_space(trial):
            return {
                "learning_rate": trial.suggest_float(
                    "learning_rate", 1e-7, 1e-4, log=True
                ),
                "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.3),
            }
        
        #def compute_objective(metrics):
        #    return best_loss_callback.best_eval_loss
        
        best_run = trainer.hyperparameter_search(
            hp_space=hp_space,
            backend="optuna",
            n_trials=10,
            study_name=f"sfam_hyperparameter_search_{time.strftime('%Y-%m-%d_%H-%M-%S')}",
            storage=f"sqlite:///sfam_{HPO_FILE}",
            compute_objective=lambda metrics: metrics["eval_loss"],
            direction="minimize",
                )
        if isinstance(best_run, list):
            best_run = best_run[0]
        print(
            f"Best hyperparameters for SFAM: learning_rate: {best_run.hyperparameters['learning_rate']}, weight_decay: {best_run.hyperparameters['weight_decay']}"
        )
        for k, v in best_run.hyperparameters.items():
            setattr(trainer.args, k, v)
        trainer.train()

    else:
        # Train the model
        trainer.train()

    # Save the model
    trainer.save_model(f"{MODEL_OUTPUT_FOLDER}sfam")

    trained_model = trainer.model

    # Ensure model is in evaluation mode
    trained_model.eval()

    log_time(message="Training SFAM", start_time=timer, logger=logger)
    test_timer = time.time()

    # Perform inference
    all_preds = []
    all_labels = []
    for batch in dataset["validation"]:
        input_ids = batch["input_ids"].unsqueeze(0).to("cuda")  # Add batch dimension
        attention_mask = batch["attention_mask"].unsqueeze(0).to("cuda")
        label = batch["label"]

        with torch.no_grad():
            outputs = trained_model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            prediction = torch.round(logits).item()

        all_preds.append(prediction)
        all_labels.append(label)

    # Compute accuracy & F1-score
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    
    if do_hyperparameter_optimization:
        print(
                f"Best hyperparameters for SFAM: learning_rate: {best_run.hyperparameters['learning_rate']}, weight_decay: {best_run.hyperparameters['weight_decay']}"
            )
    print(f"SFAM - Val loss: {best_loss_callback.best_eval_loss}, Val Accuracy: {accuracy:.4f}, Val F1-score: {f1:.4f}")

    log_time(
        message="validating SFAM", start_time=test_timer, logger=logger, level="debug"
    )
    if accuracy < SFAM_MIN_ACCURACY:
        logger.error(
            f"SFAM val accuracy of {accuracy} is below the threshold of {SFAM_MIN_ACCURACY}"
        )
        raise ValueError(
            f"SFAM val accuracy of {accuracy} is below the threshold of {SFAM_MIN_ACCURACY}"
        )
    shutil.rmtree("sfam_data", ignore_errors=True)
    return accuracy, f1


def _prepare_data_sentences(
    max_num_samples: None | dict[DATA_TYPE_ENUM, int] = None,
) -> DatasetDict:
    def get_max_num_samples(data_type: DATA_TYPE_ENUM) -> int | None:
        if max_num_samples is None or data_type not in max_num_samples:
            return None
        return max_num_samples[data_type]

    ds = DatasetDict(
        {
            "train": Dataset.from_pandas(
                _get_dataset_restricted_to_sentences_of_accepted_clusters(
                    "sfam_train",
                    shuffle=True,
                    max_num_samples=get_max_num_samples("sfam_train"),
                )
            ),
            "validation": Dataset.from_pandas(
                _get_dataset_restricted_to_accepcted_clusters(
                    "sfam_val", max_num_samples=get_max_num_samples("sfam_val")
                )
            ),
            "test": Dataset.from_pandas(
                _get_dataset_restricted_to_accepcted_clusters(
                    "sfam_test", max_num_samples=get_max_num_samples("sfam_test")
                )
            )
        }
    )
    tokenizer = AutoTokenizer.from_pretrained(SFAM_MODEL, use_fast=False)

    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=INPUT_MAX_LENGTH,
        )

    tokenized_datasets = ds.map(tokenize_function, batched=True)

    # Split the dataset into train and test
    train_dataset = tokenized_datasets["train"].remove_columns(["text"])
    eval_dataset = tokenized_datasets["validation"].remove_columns(["text"])
    test_dataset = tokenized_datasets["test"]

    # Convert dataset to torch tensors
    train_dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "label"]
    )
    eval_dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "label"]
    )
    test_dataset.set_format(
        type="torch", columns=["text", "input_ids", "attention_mask", "label"]
    )

    logger.debug(
        f"SFAM Training dataset: {len(train_dataset)}, validation dataset: {len(eval_dataset)}, test dataset: {len(test_dataset)}"
    )
    return DatasetDict(
        {"train": train_dataset, "validation": eval_dataset, "test": test_dataset}
    )

def _get_dataset_restricted_to_sentences_of_accepted_clusters(
    data_type: str, shuffle=False, max_num_samples: Optional[int] = None
) -> pd.DataFrame:
     # TODO: use the sentences with the largest range between positive and negative sample
    database = Database()
    timer = time.time()
    answers = database.get_answers_with_sentence_ids([data_type])
    
    if "val" in data_type or "test" in data_type:
        exit("ERROR Wrong data type!")
    else:
        NUM_TOP_ANSWERS = int(len(answers) * 0.01)
        all_sentences = database.get_all_style_sentences_of_accepted_clusters([data_type])    
        accepted_clusters = database.get_style_vector()
        similarities = SentenceTransformerModel.similarities(
            all_sentences["embedding"].values.tolist(),
            accepted_clusters["embedding"].values.tolist(),
        )
        answer_similarities = []
        for _, row in tqdm(
            answers.iterrows(),
            total=len(answers),
            desc=f"Calculating similarities for all answers ({data_type})",
            smoothing=0,
        ):
            data_vector: np.ndarray = row["style_vector_data"]
            data_vector = data_vector.astype(np.int32)
            sims = similarities[:, data_vector == 1]
            sims = np.nan_to_num(np.mean(sims, axis=1), nan=0.0)
            answer_similarities.append(sims)
        # an array of num of answers x num_style_sentences
        # the avg similarity of the style_vector_attributes of the answer to each style_sentence
        answer_similarities = np.array(answer_similarities)
        dataset_list = []
        for index, row in tqdm(
            all_sentences.iterrows(),
            total=len(all_sentences),
            desc=f"Creating dataset for all style_sentences ({data_type})",
            smoothing=0,
        ):
            # the similarity of all style_vector_attributes of the answer to the current style_sentence
            current_similarities = answer_similarities[:, index]
            sorted_similarity_indeces = np.argsort(current_similarities)[::-1]
            # the sentence has to be used to describe the answer; this does not necessarily apply to the top answer
            # if the top 10 answers do not include any answer with the current
            # style_sentence, the sentence is not used for the training
            current_positives = answers.iloc[sorted_similarity_indeces].head(
                NUM_TOP_ANSWERS
            )
            # the sentence must not be used to describe the answer; this applies to most answers
            current_negatives = answers.iloc[sorted_similarity_indeces].tail(
                NUM_TOP_ANSWERS
            )
            current_positives = current_positives[
                current_positives["sentence_ids"].apply(lambda x: row["id"] in x)
            ].head(max_num_samples)
            current_negatives = current_negatives[
                current_negatives["sentence_ids"].apply(lambda x: row["id"] not in x)
            ].tail(len(current_positives))
            
            # if current_positives is empty, current_negatives is also empty and the sentence is not used
            if len(current_positives) == len(current_negatives):
                current_positives["label"] = 1.0
                current_negatives["label"] = 0.0
                current_positives["text"] = current_positives["answer"].apply(
                    lambda x: _format_sfam_input(row["style_sentence"], x)
                )
                current_negatives["text"] = current_negatives["answer"].apply(
                    lambda x: _format_sfam_input(row["style_sentence"], x)
                )
                
                dataset_list.extend(current_positives[["text", "label"]].values.tolist())
                dataset_list.extend(current_negatives[["text", "label"]].values.tolist())
                
        dataset = pd.DataFrame(
            dataset_list,
            columns=["text", "label"],
        )
    #print(dataset)
    if shuffle:
        dataset = dataset.sample(frac=1)
    dataset = dataset.reset_index(drop=True)
    log_time(
        message=f"Creating dataset for {data_type}",
        start_time=timer,
        logger=logger,
        level="debug",
    )
    return dataset


def _get_dataset_restricted_to_accepcted_clusters(
    data_type: str, shuffle=False, max_num_samples: Optional[int] = None
) -> pd.DataFrame:
    database = Database()
    timer = time.time()
    answers = database.get_answers_with_sentence_ids([data_type])

    accepted_clusters = database.get_style_vector()
    similarities = SentenceTransformerModel.similarities(
        accepted_clusters["embedding"].values.tolist(),
        accepted_clusters["embedding"].values.tolist(),
    )
    answer_similarities = []
    style_vector_indices = []
    for idx, row in tqdm(
        answers.iterrows(),
        total=len(answers),
        desc=f"Calculating similarities for all answers ({data_type})",
        smoothing=0,
    ):
        data_vector: np.ndarray = row["style_vector_data"]
        data_vector = data_vector.astype(np.int32)
        sims = similarities[:, data_vector == 1]
        sims = np.nan_to_num(np.mean(sims, axis=1), nan=0.0)
        answer_similarities.append(sims)
        style_vector_indices.append(np.where(data_vector == 1)[0].tolist())
    
    # an array of num of answers x num_style_sentences
    # the avg similarity of the style_vector_attributes of the answer to each style_sentence
    answer_similarities = np.array(answer_similarities)
    answers["style_vector_indices"] = style_vector_indices

    dataset_list = []
    not_used_indices = []
    for index, row in tqdm(
        accepted_clusters.iterrows(),
        total=len(accepted_clusters),
        desc=f"Creating dataset for all style_sentences ({data_type})",
        smoothing=0,
    ):
        # the similarity of all style_vector_attributes of the answer to the current style_sentence
        current_similarities = answer_similarities[:, row["style_vector_index"]]
        sorted_similarity_indeces = np.argsort(current_similarities)[::-1]

        current_positives = answers.iloc[sorted_similarity_indeces][
            answers.iloc[sorted_similarity_indeces]["style_vector_indices"].apply(lambda x: row["style_vector_index"] in x)
        ].head(max_num_samples)

        current_negatives = answers.iloc[sorted_similarity_indeces][
            answers.iloc[sorted_similarity_indeces]["style_vector_indices"].apply(lambda x: row["style_vector_index"] not in x)
        ].tail(len(current_positives))

        if len(current_positives) == len(current_negatives):

            current_positives["label"] = 1.0
            current_negatives["label"] = 0.0
            current_positives["text"] = current_positives["answer"].apply(
                lambda x: _format_sfam_input(row["style_sentence"], x)
            )
            current_negatives["text"] = current_negatives["answer"].apply(
                lambda x: _format_sfam_input(row["style_sentence"], x)
            )

            dataset_list.extend(current_positives[["text", "label"]].values.tolist())
            dataset_list.extend(current_negatives[["text", "label"]].values.tolist())
        else:
            not_used_indices.append(row["style_vector_index"])
    print(f"WARNING Style vector indices {not_used_indices} not used!")
    
    dataset = pd.DataFrame(
        dataset_list,
        columns=["text", "label"],
    )

    #print(dataset)
    if shuffle:
        dataset = dataset.sample(frac=1)
    dataset = dataset.reset_index(drop=True)
    log_time(
        message=f"Creating dataset for {data_type}",
        start_time=timer,
        logger=logger,
        level="debug",
    )
    return dataset

def _get_dataset(
    data_type: str, shuffle=False, max_num_samples: Optional[int] = None
) -> pd.DataFrame:
    # TODO: use the sentences with the largest range between positive and negative sample
    database = Database()
    timer = time.time()
    answers = database.get_answers_with_sentence_ids([data_type])
    
    if "val" in data_type or "test" in data_type:
        accepted_clusters = database.get_style_vector()
        dataset_list_positive = []
        dataset_list_negative = []
        for _, row in tqdm(
                answers.iterrows(),
                total=len(answers),
                desc=f"Creating dataset for all answers ({data_type})",
                smoothing=0,
            ):
                data_vector: np.ndarray = row["style_vector_data"]
                for style_vector_index, label in enumerate(data_vector):
                    style_sentence = accepted_clusters.loc[accepted_clusters["style_vector_index"] == style_vector_index, "style_sentence" ].iloc[0]

                    text = _format_sfam_input(style_sentence, row["answer"])
                    sample = {"text": text, "label": label}
                    #print(sample)
                    if label == 1.0:
                        dataset_list_positive.append(sample)
                    else:
                        dataset_list_negative.append(sample)
        if max_num_samples:
            random.seed(42)   
            dataset_list_positive = random.sample(dataset_list_positive, max_num_samples // 2)
            random.seed(42)   
            dataset_list_negative = random.sample(dataset_list_negative, max_num_samples // 2)
        dataset_list = dataset_list_positive + dataset_list_negative
        dataset = pd.DataFrame(dataset_list)
    else:
        NUM_TOP_ANSWERS = int(len(answers) * 0.01)
        all_sentences = database.get_all_style_sentences([data_type])
        # shuffle, because the loop breaks if enough training samples have been found
        # without shuffle, the first sentences would always be used
        all_sentences = all_sentences.sample(frac=1, random_state=42).reset_index(drop=True)
        accepted_clusters = database.get_style_vector()
        similarities = SentenceTransformerModel.similarities(
            all_sentences["embedding"].values.tolist(),
            accepted_clusters["embedding"].values.tolist(),
        )
        answer_similarities = []
        for _, row in tqdm(
            answers.iterrows(),
            total=len(answers),
            desc=f"Calculating similarities for all answers ({data_type})",
            smoothing=0,
        ):
            data_vector: np.ndarray = row["style_vector_data"]
            data_vector = data_vector.astype(np.int32)
            sims = similarities[:, data_vector == 1]
            sims = np.nan_to_num(np.mean(sims, axis=1), nan=0.0)
            answer_similarities.append(sims)
        # an array of num of answers x num_style_sentences
        # the avg similarity of the style_vector_attributes of the answer to each style_sentence
        answer_similarities = np.array(answer_similarities)
        dataset_list = []
        for index, row in tqdm(
            all_sentences.iterrows(),
            total=len(all_sentences),
            desc=f"Creating dataset for all style_sentences ({data_type})",
            smoothing=0,
        ):
            # the similarity of all style_vector_attributes of the answer to the current style_sentence
            current_similarities = answer_similarities[:, index]
            sorted_similarity_indeces = np.argsort(current_similarities)[::-1]
            # the sentence has to be used to describe the answer; this does not necessarily apply to the top answer
            # if the top 10 answers do not include any answer with the current
            # style_sentence, the sentence is not used for the training
            current_positives = answers.iloc[sorted_similarity_indeces].head(
                NUM_TOP_ANSWERS
            )
            # the sentence must not be used to describe the answer; this applies to most answers
            current_negatives = answers.iloc[sorted_similarity_indeces].tail(
                NUM_TOP_ANSWERS
            )
            current_positives = current_positives[
                current_positives["sentence_ids"].apply(lambda x: row["id"] in x)
            ].head(1)
            current_negatives = current_negatives[
                current_negatives["sentence_ids"].apply(lambda x: row["id"] not in x)
            ].tail(len(current_positives))
            
            # if current_positives is empty, current_negatives is also empty and the sentence is not used
            if len(current_positives) == len(current_negatives):
                current_positives["label"] = 1.0
                current_negatives["label"] = 0.0
                current_positives["text"] = current_positives["answer"].apply(
                    lambda x: _format_sfam_input(row["style_sentence"], x)
                )
                current_negatives["text"] = current_negatives["answer"].apply(
                    lambda x: _format_sfam_input(row["style_sentence"], x)
                )
                
                if (
                    max_num_samples is not None
                    and len(dataset_list) + len(current_positives) + len(current_negatives)
                    > max_num_samples
                ):
                    break
                dataset_list.extend(current_positives[["text", "label"]].values.tolist())
                dataset_list.extend(current_negatives[["text", "label"]].values.tolist())
                
        dataset = pd.DataFrame(
            dataset_list,
            columns=["text", "label"],
        )

    if max_num_samples is not None and len(dataset) < max_num_samples:
        logger.warning(
            f"{data_type} has {len(dataset)} samples, less than {max_num_samples}"
        )
    #print(dataset)
    if shuffle:
        dataset = dataset.sample(frac=1)
    dataset = dataset.reset_index(drop=True)
    log_time(
        message=f"Creating dataset for {data_type}",
        start_time=timer,
        logger=logger,
        level="debug",
    )
    return dataset


def delete_model():
    database = Database()
    timer = time.time()
    shutil.rmtree(f"{MODEL_OUTPUT_FOLDER}sfam", ignore_errors=True)
    shutil.rmtree(f"{MODEL_OUTPUT_FOLDER}checkpoints/sfam", ignore_errors=True)
    database.remove_style_vector("sfam")
    log_time(message="Deleting SFAM model", start_time=timer, logger=logger)


def _format_sfam_input(style_sentence: str, answer: str):
    return f"{style_sentence} ||| {answer}"
