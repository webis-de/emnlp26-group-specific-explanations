import os
import shutil
from typing import List, Literal
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from safetensors.torch import load_file
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import (
    Trainer,
    TrainingArguments,
    EvalPrediction,
)
import logging
import time
import gc

from utils.database import Database
from config_new_prompts import (
    HPO_FILE,
    MODEL_OUTPUT_FOLDER,
)
from helper import log_time, HyperparameterSearchLogger

logger = logging.getLogger(__name__)


class StyleEmbeddingModel(nn.Module):
    def __init__(
        self,
        loss_distance: Literal["cosine", "euclidean"] = "euclidean",
        loss_margin=2.0,
    ):
        super().__init__()
        database = Database()
        style_vector_df = database.get_style_vector()
        self.fc = nn.Linear(len(style_vector_df), len(style_vector_df))
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(len(style_vector_df), len(style_vector_df))
        if loss_distance == "cosine":
            self.loss = nn.TripletMarginWithDistanceLoss(
                distance_function=lambda x, y: 1.0
                - nn.functional.cosine_similarity(x, y, dim=0),
                margin=loss_margin,
            )
        elif loss_distance == "euclidean":
            self.loss = nn.TripletMarginLoss(margin=loss_margin)
        else:
            raise ValueError(
                f"Invalid loss_distance: {loss_distance}. Must be 'cosine' or 'euclidean'."
            )

    def forward_fn(self, x):
        return self.fc2(self.relu(self.fc(x)))

    def forward(self, x=None, anchor=None, positive=None, negative=None, **kwargs):
        # If a single tensor x is provided, use it for the forward pass.
        if x is not None:
            return self.forward_fn(x)
        # If triplet inputs are provided via keywords, compute embeddings for each.
        elif anchor is not None and positive is not None and negative is not None:
            anchor_emb = self.forward_fn(anchor)
            positive_emb = self.forward_fn(positive)
            negative_emb = self.forward_fn(negative)
            # Returning a dict is one option; adjust based on how you want to evaluate.
            return {
                "anchor": anchor_emb,
                "positive": positive_emb,
                "negative": negative_emb,
                "loss": self.loss(anchor_emb, positive_emb, negative_emb),
            }
        else:
            raise ValueError("Invalid input provided to forward method.")


class CustomTrainerArguments(TrainingArguments):
    loss_distance: Literal["cosine", "euclidean"] = "euclidean"
    loss_margin: float = 2.0

    def __init__(self, **kwargs):
        # remove custom arguments from kwargs before calling super
        self.loss_distance = kwargs.pop("loss_distance", "euclidean")
        self.loss_margin = kwargs.pop("loss_margin", 2.0)
        super().__init__(**kwargs)


def train_and_create_style_embedding(
    recompute=False, do_hyperparameter_optimization=False
):
    if recompute:
        delete_model()

    _train(do_hyperparameter_optimization=do_hyperparameter_optimization)
    _create_style_embedding(f"{MODEL_OUTPUT_FOLDER}embedder")

    
def train(
    recompute=False, do_hyperparameter_optimization=False
):
    if recompute:
        delete_model()

    _train(do_hyperparameter_optimization=do_hyperparameter_optimization)
    
def predict_style_embedding():
    _create_style_embedding(f"{MODEL_OUTPUT_FOLDER}embedder")

def _create_style_embedding(model_path: str):
    database = Database()
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
    #answers = answers[answers["style_embedding"].isnull()]
    if len(answers) == 0:
        print("Style embeddings already created; skipping")
        return
    answers = answers.rename(columns={"style_vector_lisa": "style_vector"})
    embeddings = create_style_embedding(model_path, answers)
    database.update_answer_styleEmbedding(embeddings)


def create_style_embedding(model_path: str, data: pd.DataFrame) -> pd.DataFrame:
    model = StyleEmbeddingModel()
    model.load_state_dict(load_file(f"{model_path}/model.safetensors"))
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    timer = time.time()
    style_vectors = np.array(data["style_vector"].values.tolist())
    
    style_vectors = torch.tensor(style_vectors).to(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    with torch.no_grad():
        style_embeddings = model(style_vectors)
    style_embeddings = style_embeddings.cpu().numpy().astype(np.float32)
    result = pd.DataFrame(
        {
            "answer_id": data["id"],
            "style_embedding": [e for e in style_embeddings],
        }
    )
    log_time("Creating style embeddings", start_time=timer, logger=logger)
    del model
    torch.cuda.empty_cache()
    gc.collect()
    return result


def _train(do_hyperparameter_optimization):
    if os.path.exists(f"{MODEL_OUTPUT_FOLDER}embedder"):
        print("Embedding model already exists, skipping training.")
        return
    dataset = _prepare_data()
    timer = time.time()

    def model_init(training_args: CustomTrainerArguments):
        return StyleEmbeddingModel(
            loss_distance=training_args.loss_distance,
            loss_margin=training_args.loss_margin,
        )

    # Set up training arguments
    training_args = CustomTrainerArguments(
        output_dir=f"{MODEL_OUTPUT_FOLDER}checkpoints/embedder",  # output directory for model predictions and checkpoints
        save_total_limit=2,
        per_device_train_batch_size=12,
        per_device_eval_batch_size=300,
        eval_strategy="steps",
        save_strategy="steps",
        logging_strategy="steps",
        num_train_epochs=10,  # number of training epochs
        logging_steps=500, 
        save_steps=500,
        report_to="none",
        remove_unused_columns=False,  # Necessary for triplet loss
        label_names=["labels"],  # Necessary for triplet loss
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        # loss_distance="cosine",
        # loss_margin=0.2,
        loss_distance="euclidean",
        loss_margin=2.0,
    )

    # Initialize the trainer with the validation dataset as eval_dataset
    trainer = Trainer(
        model_init=lambda: model_init(training_args),
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset[
            "validation"
        ],  # This will be used for validation during training
        compute_metrics=_create_compute_metrics_fn(training_args),
        callbacks=[HyperparameterSearchLogger(logger)],
    )

    if do_hyperparameter_optimization:

        def hp_space(trial):
            return {
                "learning_rate": trial.suggest_float(
                    "learning_rate", 1e-7, 1e-4, log=True
                ),
                "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.3)
            }

        best_run = trainer.hyperparameter_search(
            hp_space=hp_space,
            backend="optuna",
            n_trials=30,
            study_name=f"embedder_hyperparameter_search_{time.strftime('%Y-%m-%d_%H-%M-%S')}",
            storage=f"sqlite:///embedder_{HPO_FILE}",
            compute_objective=lambda metrics: metrics["eval_loss"],
            direction="minimize",
            # compute_objective=lambda metrics: metrics["eval_loss"],
            # direction="minimize",
        )
        if isinstance(best_run, list):
            best_run = best_run[0]
        print(
            f"Best hyperparameters for embedder: learning_rate: {best_run.hyperparameters['learning_rate']}, weight_decay: {best_run.hyperparameters['weight_decay']}"
        )
        for k, v in best_run.hyperparameters.items():
            setattr(trainer.args, k, v)
        trainer.train()

    else:
        # Train the model
        trainer.train()

    log_time("Training the embedding model", start_time=timer, logger=logger)

    # Save the trained model
    trainer.save_model(f"{MODEL_OUTPUT_FOLDER}embedder")

    # Evaluate on the test dataset after training
    trained_model = trainer.model
    trained_model.to("cuda" if torch.cuda.is_available() else "cpu")
    trained_model.eval()

    # Initialize metrics
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    test_data = DataLoader(dataset["validation"], batch_size=16)

    # Disable gradient calculation for evaluation
    with torch.no_grad():
        for batch in test_data:
            anchor = batch["anchor"].to("cuda" if torch.cuda.is_available() else "cpu")
            positive = batch["positive"].to(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            negative = batch["negative"].to(
                "cuda" if torch.cuda.is_available() else "cpu"
            )

            # Compute embeddings using the trained_model's forward method
            anchor_emb = trained_model(anchor)
            positive_emb = trained_model(positive)
            negative_emb = trained_model(negative)

            # Compute the triplet loss for the current batch
            loss = trained_model.loss(anchor_emb, positive_emb, negative_emb)
            total_loss += loss.item() * anchor.size(0)

            # Calculate distances (Euclidean norm)
            # pos_dist = torch.norm(anchor_emb - positive_emb, p=2, dim=1)
            # neg_dist = torch.norm(anchor_emb - negative_emb, p=2, dim=1)
            pos_dist = 1.0 - nn.functional.cosine_similarity(
                anchor_emb, positive_emb, dim=1
            )
            neg_dist = 1.0 - nn.functional.cosine_similarity(
                anchor_emb, negative_emb, dim=1
            )

            # Count how many triplets are "correct" (i.e., positive pair is closer than negative pair)
            correct = (pos_dist < neg_dist).sum().item()
            total_correct += correct
            total_samples += anchor.size(0)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    if do_hyperparameter_optimization:
        print(
            f"Best hyperparameters for embedder: learning_rate: {best_run.hyperparameters['learning_rate']}, weight_decay: {best_run.hyperparameters['weight_decay']}"
        )
    print(
        f"Custom Test Results: Average Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}"
    )


def _create_compute_metrics_fn(training_args: CustomTrainerArguments):
    print(
        f"Setting up metrics function for embedding model with loss_distance: {training_args.loss_distance}, loss_margin: {training_args.loss_margin}"
    )

    def compute_metrics(eval_pred: EvalPrediction):
        """
        Expects eval_pred to be a tuple (predictions, label_ids), where
        predictions is a tuple with the np.arrays anchor, positive and negative.
        """
        predictions, _ = (
            eval_pred  # We ignore label_ids since our dataset is unsupervised for this loss.
        )
        predictions = [torch.tensor(p) for p in predictions]
        if training_args.loss_distance == "cosine":
            triplet_loss = nn.TripletMarginWithDistanceLoss(
                distance_function=lambda x, y: 1.0
                - nn.functional.cosine_similarity(x, y, dim=0),
                margin=training_args.loss_margin,
            )(*predictions)
            pos_dist = 1.0 - nn.functional.cosine_similarity(
                predictions[0], predictions[1], dim=1
            )
            neg_dist = 1.0 - nn.functional.cosine_similarity(
                predictions[0], predictions[2], dim=1
            )
        else:
            triplet_loss = nn.TripletMarginLoss(margin=training_args.loss_margin)(
                *predictions
            )
            pos_dist = torch.norm(predictions[0] - predictions[1], p=2, dim=1)
            neg_dist = torch.norm(predictions[0] - predictions[2], p=2, dim=1)
        accuracy = (pos_dist < neg_dist).float().mean()
        return {"loss": triplet_loss, "accuracy": accuracy}

    return compute_metrics


def _prepare_data() -> DatasetDict:
    def get_dataset(data_types: List[str], shuffle=False) -> Dataset:
        database = Database()
        answers = database.get_answers(data_types)

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
            anchor_vectors = group["style_vector_lisa"].values.tolist()
            positive_vectors = positive["style_vector_lisa"].values.tolist()
            negative_vectors = negative["style_vector_lisa"].values.tolist()
            result.extend(
                [
                    [a, p, z]
                    for (a, p, z) in zip(
                        anchor_vectors, positive_vectors, negative_vectors
                    )
                ]
            )
        result_df = pd.DataFrame(result, columns=["anchor", "positive", "negative"])
        if shuffle:
            result_df = result_df.sample(frac=1).reset_index(drop=True)
        # add a dummy label column so the compute_metric function is called
        result_df["labels"] = 0.0
        dataset = Dataset.from_pandas(result_df)
        dataset.set_format(
            type="torch", columns=["anchor", "positive", "negative", "labels"]
        )
        return dataset

    ds = DatasetDict(
        {
            "train": get_dataset(
                ["sfam_train", "sfam_val", "sfam_test", "lisa_train"], shuffle=True
            ),
            "validation": get_dataset(["lisa_val"]),
            "test": get_dataset(["lisa_test"]),
        }
    )
    logger.debug(
        f"Embedder training dataset: {len(ds['train'])}, validation dataset: {len(ds['validation'])}, test dataset: {len(ds['test'])}"
    )
    return ds


def delete_model():
    database = Database()
    shutil.rmtree(f"{MODEL_OUTPUT_FOLDER}embedder", ignore_errors=True)
    shutil.rmtree(f"{MODEL_OUTPUT_FOLDER}checkpoints/embedder", ignore_errors=True)
    database.remove_style_embedding()
