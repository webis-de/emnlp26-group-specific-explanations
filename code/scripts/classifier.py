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
from sklearn.metrics import accuracy_score, f1_score

from utils.database import Database
from config_new_prompts import (
    HPO_FILE,
    MODEL_OUTPUT_FOLDER,
)
from helper import log_time, HyperparameterSearchLogger
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


class GroupSpecificityClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 1024),
            #nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(0.2),

            nn.Linear(1024, 512),
            #nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.2),

            nn.Linear(512, 256),
            #nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.1),

            nn.Linear(256, num_classes),
        )
        self.loss = nn.CrossEntropyLoss()

    def forward(self, attribute_vectors, labels=None):
        logits = self.model(attribute_vectors)
        
        # Compute loss if labels are provided
        loss = None
        if labels is not None:
            loss = self.loss(logits, labels)

        return {"loss": loss, "logits": logits}
    
def train(
    recompute=False, do_hyperparameter_optimization=False
):
    if recompute:
        delete_model()

    _train(do_hyperparameter_optimization=do_hyperparameter_optimization)
    
def predict_fit_score(data: pd.DataFrame) -> pd.DataFrame:
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
    timer = time.time()
    style_vectors = data["style_vector"].values.tolist()
    target_groups = data["data_group"].values.tolist()
    
    labels = []
    for group in target_groups:
        group_index = groups.index(group)
        labels.append(group_index)
        
    
    labels = torch.tensor(np.array(labels))
    
    style_vectors = torch.tensor(np.array(style_vectors)).to(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    # Centering
    style_vectors = style_vectors - 0.5
    
    with torch.no_grad():
        logits = model(style_vectors)["logits"].cpu()

    # Convert logits to probabilities
    probs = torch.softmax(logits, dim=-1)
    # Predicted class
    #pred_classes = torch.argmax(probs, dim=-1)

    # Fit scores of ground-truth class
    fit_scores = probs.gather(1, labels.unsqueeze(1))
    fit_scores = fit_scores.flatten()
    

    #accuracy = accuracy_score(labels, pred_classes.cpu().numpy())
    
    del model
    torch.cuda.empty_cache()
    gc.collect()
    return fit_scores


def _train(do_hyperparameter_optimization):
    if os.path.exists(f"{MODEL_OUTPUT_FOLDER}classifier"):
        print("Classifier model already exists, skipping training.")
        return
    
    database = Database()
    style_vector_df = database.get_style_vector()
    groups = sorted(database.get_groups(
        ["sfam_train", "sfam_val", "sfam_test", "lisa_train", "lisa_val", "lisa_test"]
    ))
    
    dataset = _prepare_data(groups)
    timer = time.time()

    # compute loss function that provides the accuracy
    def compute_metrics(pred: EvalPrediction):
        # Convert logits to torch tensor
        logits = torch.tensor(pred.predictions)

        # Convert logits to probabilities
        probs = torch.softmax(logits, dim=-1)

        # Predicted class
        pred_classes = torch.argmax(probs, dim=-1)
        
        # Mean fit score of ground-truth class
        fit_score = probs.gather(1, torch.tensor(pred.label_ids).unsqueeze(1)).mean().item()

        accuracy = accuracy_score(pred.label_ids, pred_classes.cpu().numpy())

        print(f"Classifier validation accuracy: {accuracy}, validation fit score: {fit_score}", flush=True)
        return {
            "accuracy": accuracy,
            "fit_score": fit_score,
        }
    
    # Define TrainingArguments
    training_args = TrainingArguments(
        output_dir=f"{MODEL_OUTPUT_FOLDER}checkpoints/classifier",  # output directory for model predictions and checkpoints
        eval_strategy="steps",
        save_strategy="steps",
        logging_strategy="steps",
        num_train_epochs=60,  # number of training epochs
        learning_rate=5e-6,
        weight_decay=0.15,
        logging_steps=100, 
        save_steps=100,
        per_device_train_batch_size=128,  # batch size per device during training
        per_device_eval_batch_size=1000,  # batch size per device during evaluation
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="eval_fit_score",
        greater_is_better=True,
    )

    # Initialize Trainer
    trainer = Trainer(
        model_init=lambda: GroupSpecificityClassifier(input_dim=len(style_vector_df), num_classes=len(groups)).to("cuda" if torch.cuda.is_available() else "cpu"),
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        callbacks=[HyperparameterSearchLogger(logger)],
        compute_metrics=compute_metrics,
    )

    if do_hyperparameter_optimization:

        def hp_space(trial):
            return {
                "learning_rate": trial.suggest_float(
                    "learning_rate", 1e-5, 1e-2, log=True
                ),
                "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.3),
            }

        best_run = trainer.hyperparameter_search(
            hp_space=hp_space,
            compute_objective=lambda metrics: metrics["eval_fit_score"],
            direction="maximize",
            backend="optuna",
            n_trials=100,
            study_name=f"classifier_hyperparameter_search_{time.strftime('%Y-%m-%d_%H-%M-%S')}",
            storage=f"sqlite:///classifier_{HPO_FILE}",
        )
        if isinstance(best_run, list):
            best_run = best_run[0]
        print(
            f"Best hyperparameters for Classifier: learning_rate: {best_run.hyperparameters['learning_rate']}, weight_decay: {best_run.hyperparameters['weight_decay']}"
        )
        for k, v in best_run.hyperparameters.items():
            setattr(trainer.args, k, v)
        trainer.train()
    else:
        # Train the model
        trainer.train()
    trainer.save_model(f"{MODEL_OUTPUT_FOLDER}classifier")

    trained_model = trainer.model

    # Ensure model is in evaluation mode
    trained_model.eval()

    predictions = []
    fit_scores = []
    ground_truths = []

    device = "cuda" if torch.cuda.is_available() else "cpu"
    trained_model.to(device)
    trained_model.eval()

    for sample in dataset["validation"]:
        attribute_vectors = sample["attribute_vectors"].to(device)
        labels = sample["labels"].to(device)

        with torch.no_grad():
            outputs = trained_model(attribute_vectors)
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
    
    if do_hyperparameter_optimization:
        print(
                f"Best hyperparameters for Classifier: learning_rate: {best_run.hyperparameters['learning_rate']}, weight_decay: {best_run.hyperparameters['weight_decay']}"
            )
    print(f"Classifier - Val Accuracy: {accuracy:.4f}, Val F1-score: {f1:.4f}, Val Fit Score: {mean_fit_score:.4f}")

def _prepare_data(groups) -> DatasetDict:
    database = Database()

    def get_dataset(data_types: List[str], shuffle=False) -> pd.DataFrame:
        answers = database.get_answers(data_types)
        
        labels = []
        answer_attribute_vectors = []
        for index, row in answers.iterrows():
            group_index = groups.index(row["data_group"])
            labels.append(group_index)
            answer_attribute_vectors.append(row["style_vector_lisa"])
        
        # Convert to tensor for normalization
        X = torch.tensor(answer_attribute_vectors, dtype=torch.float)
        # Centering
        X = X - 0.5
        # Convert back to list for storing in dataframe
        answer_attribute_vectors = X.tolist()
        
        dataset = pd.DataFrame({
            "attribute_vectors": answer_attribute_vectors,
            "labels": labels
        })
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

    # Convert dataset to torch tensors
    for split in ["train", "validation", "test"]:
        ds[split].set_format(
            type="torch", columns=["attribute_vectors", "labels"]
        )

    print(
        f"Classifier Training dataset: {len(ds['train'])}, validation dataset: {len(ds['validation'])}, test dataset: {len(ds['test'])}"
    )
    return ds

    
#def _prepare_data(groups) -> DatasetDict:
#    database = Database()
#    
#    def get_dataset(data_types: List[str], shuffle=False, pca=None, pca_mean=None, pca_std=None, fit_pca=False) -> pd.DataFrame:
#        answers = database.get_answers(data_types)
#        
#        labels = []
#        answer_attribute_vectors = []
#        for index, row in answers.iterrows():
#            group_index = groups.index(row["data_group"])
#            labels.append(group_index)
#            answer_attribute_vectors.append(row["style_vector_lisa"])
#        #labels_tensor = torch.tensor(labels, dtype=torch.long)
#        #labels_one_hot = torch.nn.functional.one_hot(labels_tensor, num_classes=len(groups))
#        
#        # Convert to tensor for normalization
#        X = torch.tensor(answer_attribute_vectors, dtype=torch.float)
#        # Normalize features (column-wise)
#        #X = (X - X.mean(dim=0)) / X.std(dim=0)
#        # Center around mid
#        #X = X - 0.5
#        if fit_pca:
#            pca = PCA(n_components=0.99)  # keep 99% variance
#            X_pca = pca.fit_transform(X.numpy())
#            pca_mean = X_pca.mean(axis=0)
#            pca_std = X_pca.std(axis=0)
#            X_pca = (X_pca - pca_mean) / (pca_std)
#            X = torch.tensor(X_pca, dtype=torch.float)
#            # Convert back to list for storing in dataframe
#            answer_attribute_vectors = X.tolist()
#            
#        else:
#            X_pca = pca.transform(X.numpy())
#            X_pca = (X_pca - pca_mean) / (pca_std)
#            X = torch.tensor(X_pca, dtype=torch.float)
#            # Convert back to list for storing in dataframe
#            answer_attribute_vectors = X.tolist()
#        
#        dataset = pd.DataFrame({
#            "attribute_vectors": answer_attribute_vectors,
#            "labels": labels
#        })
#        if shuffle:
#            dataset = dataset.sample(frac=1)
#        dataset = dataset.reset_index(drop=True)
#        return dataset, pca, pca_mean, pca_std
#    
#    
#    train_dataset, pca, pca_mean, pca_std = get_dataset(["sfam_train", "sfam_val", "sfam_test", "lisa_train"], shuffle=True, fit_pca=True)
#    input_dim = len(train_dataset["attribute_vectors"][0])
#    print(input_dim)
#    train_dataset = Dataset.from_pandas(train_dataset)
#    
#    
#    eval_dataset, _, _, _ = get_dataset(["lisa_val"], pca=pca, pca_mean=pca_mean, pca_std=pca_std)
#    eval_dataset = Dataset.from_pandas(eval_dataset)
#    
#    test_dataset, _, _, _ = get_dataset(["lisa_test"], pca=pca, pca_mean=pca_mean, pca_std=pca_std)
#    test_dataset = Dataset.from_pandas(test_dataset)
#
#    # Convert dataset to torch tensors
#    train_dataset.set_format(
#        type="torch", columns=["attribute_vectors", "labels"]
#    )
#    eval_dataset.set_format(
#        type="torch", columns=["attribute_vectors", "labels"]
#    )
#    test_dataset.set_format(
#        type="torch", columns=["attribute_vectors", "labels"]
#    )
#
#    print(
#        f"Classifier Training dataset: {len(train_dataset)}, validation dataset: {len(eval_dataset)}, test dataset: {len(test_dataset)}"
#    )
#    return DatasetDict(
#        {"train": train_dataset, "validation": eval_dataset, "test": test_dataset}
#    ), input_dim


def delete_model():
    database = Database()
    shutil.rmtree(f"{MODEL_OUTPUT_FOLDER}classifier", ignore_errors=True)
    shutil.rmtree(f"{MODEL_OUTPUT_FOLDER}checkpoints/classifier", ignore_errors=True)
