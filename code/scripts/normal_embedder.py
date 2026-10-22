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

from utils.database import Database
from config_new_prompts import (
    HPO_FILE,
    MODEL_OUTPUT_FOLDER,
    STYLE_VECTOR_SIZE,
    STYLE_EMBEDDING_SIZE,
    STYLE_EMBEDDING_VECTOR,
)
from utils.sentence_transformer import SentenceTransformerModel
from helper import log_time, HyperparameterSearchLogger
import gc

logger = logging.getLogger(__name__)

def create_normal_embedding():
    database = Database()
    answers = database.get_answers(
        [
            "sfam_train",
            "sfam_val",
            "sfam_test",
            "lisa_train",
            "lisa_val",
            "lisa_test",
            "foreign_domain_test",
        ]
    )

    answers = answers[answers["normal_embedding"].isnull()]
    if len(answers) == 0:
        print()("Normal embeddings already created; skipping")
        return
    
    embedding_model = SentenceTransformerModel(llm_enabled=True)

    answers = np.array(data["answer"].values.tolist())
    normal_embeddings = embedding_model.encode(answers)
    normal_embeddings = normal_embeddings.astype(np.float32)
    embeddings = pd.DataFrame(
        {
            "answer_id": data["id"],
            "normal_embedding": [e for e in normal_embeddings],
        }
    )
    database.update_answer_normalEmbedding(embeddings)
    del embedding_model
    torch.cuda.empty_cache()
    gc.collect()

def get_normal_embedding(data: pd.DataFrame) -> pd.DataFrame:
    database = Database()
    BATCH_SIZE = 512
    embedding_model = SentenceTransformerModel(llm_enabled=True)
 
    answers = np.array(data["answer"].values.tolist())
    ids = np.array(data["id"].values.tolist())
    for index in range(0, len(answers), BATCH_SIZE):
        batch_answers = answers[index : index + BATCH_SIZE]
        batch_ids = ids[index : index + BATCH_SIZE]
        normal_embeddings = embedding_model.encode(batch_answers)
        normal_embeddings = normal_embeddings.astype(np.float32)
        embeddings = pd.DataFrame(
            {
                "id": batch_ids,
                "vector": [e for e in normal_embeddings],
            }
        )
        database.update_steering_generations_vector(embeddings, "normal_embedding")
    del embedding_model
    torch.cuda.empty_cache()
    gc.collect()
 