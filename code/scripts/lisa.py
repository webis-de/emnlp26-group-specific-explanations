import shutil
import time
from typing import List
import pandas as pd
import torch
import logging
import os
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    TrainerCallback
)
import torch.nn.functional as F
from tqdm import trange
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from config_new_prompts import (
    MODEL_OUTPUT_FOLDER,
    LISA_MODEL,
    STYLE_VECTOR_SIZE,
    HPO_FILE,
)
from helper import HyperparameterSearchLogger, log_time, AutoModelWithSigmoid
from utils.database import Database
from typing import Literal
import gc

logger = logging.getLogger(__name__)
INPUT_MAX_LENGTH = 512


def train_and_create_style_vector(
    recompute=False, do_hyperparameter_optimization=False
):
    if recompute:
        delete_model()

    _train(
        do_hyperparameter_optimization=do_hyperparameter_optimization,
    )
    _create_style_vector(f"{MODEL_OUTPUT_FOLDER}lisa")

    
def train(
    recompute=False, do_hyperparameter_optimization=False
):
    if recompute:
        delete_model()

    _train(do_hyperparameter_optimization=do_hyperparameter_optimization)
    
def predict_style_vector():
    _create_style_vector(f"{MODEL_OUTPUT_FOLDER}lisa", batch)

def _train(
    do_hyperparameter_optimization: bool = False,
) -> tuple[float, float, float]:
    if os.path.exists(f"{MODEL_OUTPUT_FOLDER}lisa"):
        print("Lisa model already exists, skipping training.")
        return
    
    dataset = _prepare_data_sentences()
    timer = time.time()
    
    database = Database()
    style_vector_df = database.get_style_vector()

    # Define TrainingArguments
    training_args = TrainingArguments(
        output_dir=f"{MODEL_OUTPUT_FOLDER}checkpoints/lisa",  # output directory for model predictions and checkpoints
        eval_strategy="steps",
        save_strategy="steps",
        logging_strategy="steps",
        num_train_epochs=5,  # number of training epochs
        learning_rate=5e-6,
        weight_decay=0.15,
        logging_steps=500, 
        save_steps=500,
        per_device_train_batch_size=12,  # batch size per device during training
        per_device_eval_batch_size=300,  # batch size per device during evaluation
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        dataloader_num_workers=8,
        dataloader_pin_memory=True,
        # default loss is MSE
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
            LISA_MODEL,
            num_labels=len(style_vector_df),
            output_hidden_states=True,
            problem_type="regression",
            dtype=torch.bfloat16
        ).to("cuda" if torch.cuda.is_available() else "cpu"),
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5, early_stopping_threshold=0.001), HyperparameterSearchLogger(logger), best_loss_callback],
    )

    if do_hyperparameter_optimization:

        def hp_space(trial):
            return {
                "learning_rate": trial.suggest_float(
                    "learning_rate", 1e-7, 1e-4, log=True
                ),
                "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.5),
            }

        best_run = trainer.hyperparameter_search(
            hp_space=hp_space,
            compute_objective=lambda metrics: metrics["eval_loss"],
            direction="minimize",
            backend="optuna",
            n_trials=20,
            study_name=f"lisa_hyperparameter_search_{time.strftime('%Y-%m-%d_%H-%M-%S')}",
            storage=f"sqlite:///{HPO_FILE}_lisa",
        )
        if isinstance(best_run, list):
            best_run = best_run[0]
        print(
            f"Best hyperparameters for LISA: learning_rate: {best_run.hyperparameters['learning_rate']}, weight_decay: {best_run.hyperparameters['weight_decay']}"
        )
        for k, v in best_run.hyperparameters.items():
            setattr(trainer.args, k, v)
        trainer.train()
    else:
        # Train the model
        trainer.train()
    trainer.save_model(f"{MODEL_OUTPUT_FOLDER}lisa")

    log_time(message="Training LISA", start_time=timer, logger=logger)

    test_timer = time.time()

    trained_model = trainer.model

    # Ensure model is in evaluation mode
    trained_model.eval()

    predictions = []
    ground_truths = []
    for batch in dataset["validation"]:
        input_ids = (
            batch["input_ids"]
            .unsqueeze(0)
            .to("cuda" if torch.cuda.is_available() else "cpu")
        )  # Add batch dimension
        attention_mask = (
            batch["attention_mask"]
            .unsqueeze(0)
            .to("cuda" if torch.cuda.is_available() else "cpu")
        )
        labels = batch["labels"].unsqueeze(0)

        with torch.no_grad():
            outputs = trained_model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions.append(logits.cpu())
            ground_truths.append(labels.cpu())

    predictions = torch.cat(predictions, dim=0)
    ground_truths = torch.cat(ground_truths, dim=0)

    # Mean Squared Error (MSE)
    mse = torch.mean((predictions - ground_truths) ** 2).item()

    # Mean Absolute Error (MAE)
    mae = torch.mean(torch.abs(predictions - ground_truths)).item()

    # Cosine Similarity (closer to 1 is better)
    cos_sim = F.cosine_similarity(predictions, ground_truths, dim=1).mean().item()

    ground_truth_predictions = ground_truths.round().flatten()
    lisa_predictions = predictions.round().flatten()
    accuracy = balanced_accuracy_score(ground_truth_predictions, lisa_predictions)
    f1 = f1_score(ground_truth_predictions, lisa_predictions, average='weighted')
    
    if do_hyperparameter_optimization:
        print(
            f"Best hyperparameters for LISA: learning_rate: {best_run.hyperparameters['learning_rate']}, weight_decay: {best_run.hyperparameters['weight_decay']}"
        )
    print(
        f"LISA - Val loss: {best_loss_callback.best_eval_loss}, Val MSE: {mse:.3f}, Val MAE: {mae:.3f}, Average Cosine Similarity: {cos_sim:.3f}, accuracy: {accuracy}, f1: {f1}"
    )

    log_time(
        message="Validating LISA", start_time=test_timer, logger=logger, level="debug"
    )

    return mse, mae, cos_sim


def _create_style_vector(model_name: str):
    database = Database()
    answer_count = database.count_answers_without_stylevector(
        vector_type="lisa",
        data_types=[
            "sfam_train",
            "sfam_val",
            "sfam_test",
            "lisa_train",
            "lisa_val",
            "lisa_test"
        ],
    )
    if answer_count == 0:
        print(
            "All answers have LISA style vectors; skipping style vector generation"
        )
        return
    
    answers = database.get_answers(
        [
            "sfam_train",
            "sfam_val",
            "sfam_test",
            "lisa_train",
            "lisa_val",
            "lisa_test"
        ]
    )
    
    print(len(answers))
    
    answers = answers[answers["style_vector_lisa"].isnull()]
    print(len(answers))
    
    result_answers = create_style_vector(model_name, answers)
    database.update_answer_styleVector(result_answers, "lisa")
    database.backup("lisa_style_vector")


def create_style_vector(model_name: str, data: pd.DataFrame) -> pd.DataFrame:
    LISA_BATCH_SIZE = 500
    timer = time.time()
    print("Started creating LISA style vector")
    tokenizer = AutoTokenizer.from_pretrained(LISA_MODEL, use_fast=False)
    model = AutoModelWithSigmoid.from_pretrained(
                model_name,
                dtype=torch.bfloat16
            )
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    input_texts = data["answer"].tolist()
    style_vectors = []
    for index in trange(
        0, len(input_texts), LISA_BATCH_SIZE, desc="Creating LISA style vectors"
    ):
        batch = input_texts[index : index + LISA_BATCH_SIZE]
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
        style_vectors.extend(logits.to(torch.float32).cpu().numpy())

    result_answers = pd.DataFrame(
        {
            "answer_id": data["id"],
            "style_vector": style_vectors,
        }
    )
    log_time(message="Creating LISA style vector", start_time=timer, logger=logger)
    del model
    torch.cuda.empty_cache()
    gc.collect()
    return result_answers


def _prepare_data_sentences() -> DatasetDict:
    database = Database()

    def get_dataset(data_types: List[str], shuffle=False) -> pd.DataFrame:
        answers = database.get_answers(data_types)
        dataset = answers[["answer", "style_vector_sfam"]]
        dataset = dataset.rename(
            columns={"answer": "text", "style_vector_sfam": "labels"}
        )
        if shuffle:
            dataset = dataset.sample(frac=1)
        dataset = dataset.reset_index(drop=True)
        return dataset

    ds = DatasetDict(
        {
            "train": Dataset.from_pandas(
                get_dataset(
                    ["sfam_train", "sfam_val", "sfam_test", "lisa_train"], shuffle=True
                )
            ),
            "validation": Dataset.from_pandas(get_dataset(["lisa_val"])),
            "test": Dataset.from_pandas(get_dataset(["lisa_test"])),
        }
    )
    tokenizer = AutoTokenizer.from_pretrained(LISA_MODEL, use_fast=False)

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
        type="torch", columns=["input_ids", "attention_mask", "labels"]
    )
    eval_dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels"]
    )
    test_dataset.set_format(
        type="torch", columns=["text", "input_ids", "attention_mask", "labels"]
    )

    logger.debug(
        f"LISA Training dataset: {len(train_dataset)}, validation dataset: {len(eval_dataset)}, test dataset: {len(test_dataset)}"
    )
    return DatasetDict(
        {"train": train_dataset, "validation": eval_dataset, "test": test_dataset}
    )


def delete_model():
    database = Database()
    timer = time.time()
    shutil.rmtree(f"{MODEL_OUTPUT_FOLDER}lisa", ignore_errors=True)
    shutil.rmtree(f"{MODEL_OUTPUT_FOLDER}checkpoints/lisa", ignore_errors=True)
    database.remove_style_vector("lisa")
    log_time(message="Deleting LISA model", start_time=timer, logger=logger)
