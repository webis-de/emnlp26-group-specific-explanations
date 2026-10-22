import logging
import time
from typing import List, Union
import numpy as np
from sentence_transformers import SentenceTransformer
from helper import log_time
import nltk  # type: ignore
from sklearn.metrics.pairwise import cosine_similarity  # type: ignore
from sklearn.neighbors import NearestNeighbors  # type: ignore
from tqdm import trange
import torch

from config_new_prompts import SENTENCE_TRANSFORMER_MODEL

logger = logging.getLogger(__name__)


class SentenceTransformerModel:
    _instance = None
    llm_enabled = False
    first_clustering = True

    def __new__(self, llm_enabled=True):
        if self._instance is None:
            self._instance = super(SentenceTransformerModel, self).__new__(self)
            if llm_enabled:
                self.llm_enabled = True
                self.first = True
        return self._instance

    def encode(self, sentences: List[str]) -> np.ndarray:
        if self.llm_enabled:
            if self.first:
                logger.debug(
                    f"Loading SentenceTransformer model ({SENTENCE_TRANSFORMER_MODEL})"
                )
                device = "cuda" if torch.cuda.is_available() else "cpu"
                print(device)
                print(SENTENCE_TRANSFORMER_MODEL)
                self.model = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL, device=device)
                
                logger.debug("Model loaded")
                self.first = False
            return self.model.encode(
                sentences, convert_to_numpy=True, show_progress_bar=False
            )
        return np.zeros((len(sentences), 384), dtype=np.float32)

    def tokenize(self, input: str) -> List[str]:
        if self.first_clustering:
            #nltk.download("punkt_tab")
            #nltk.download("punkt")
            self.first_clustering = False
        tokenized_sentences = nltk.tokenize.sent_tokenize(input)
        tokenized_sentences = [
            nltk.tokenize.line_tokenize(sentence) for sentence in tokenized_sentences
        ]
        tokenized_sentences = [
            item for sublist in tokenized_sentences for item in sublist
        ]
        return tokenized_sentences

    @staticmethod
    def similarities(
        sentences: List[np.ndarray], sentences2: Union[List[np.ndarray], None] = None
    ) -> np.ndarray:
        """Calculate the cosine similarity between all pairs of embeddings.

        Args:
            sentences (List[np.ndarray]): A list of embeddings.

        Returns:
            np.ndarray: A 2d matrix of cosine similarities.
        """
        return cosine_similarity(sentences, sentences2)

    @staticmethod
    def cluster(
        embeddings: np.ndarray,
        threshold=0.85,
    ) -> List[List[int]]:
        """Group embeddings into clusters based on similarity.

        All embeddings will be returned as the minimum cluster size is 1.

        Args:
            embeddings (np.ndarray): A two dimensional numpy array which is a list of embeddings.
            threshold (float, optional): The minimum similarity for two embeddings to belong in the same cluster. Defaults to 0.85.

        Returns:
            List[List[int]]: A list of clusters, where each cluster is a list of indices of embeddings.
            The first item is the center point of the cluster.
        """
        logger.debug(
            f"Clustering {len(embeddings):,} embeddings with threshold {threshold}"
        )
        if np.array_equal(embeddings, np.zeros((1, 384), dtype=np.float32)):
            logger.warning("Skipping clustering as there are no embeddings")
            return [[0]]
        timer = time.time()
        clusterer = NearestNeighbors(radius=1 - threshold, metric="cosine", n_jobs=-1)
        clusterer.fit(embeddings)
        neighbors = clusterer.radius_neighbors_graph(
            embeddings, mode="distance", sort_results=True
        )
        log_time(
            message="neighbor computation",
            start_time=timer,
            logger=logger,
            level="debug",
        )
        pre_cluster_timer = time.time()
        visited_ids = set()
        pre_clusters = []
        for i in trange(len(embeddings), desc="Computing pre_clusters", smoothing=0):
            if i in visited_ids:
                continue
            centroid = i
            while True:
                row = neighbors.getrow(centroid)
                current_neighbors = np.nonzero(row)[1]
                # the centroid has a distance of 0 to itself, so it is not included in np.nonzero
                # sometimes the distance is not exactly 0, so we need to check the length of the neighbors list
                if len(current_neighbors) != row.getnnz():
                    current_neighbors = np.insert(current_neighbors, 0, centroid)
                # break if the current centroid is the center of the cluster, the center is already visited or the distance to the center is practically 0
                if (
                    centroid == current_neighbors[0]
                    or current_neighbors[0] in visited_ids
                    or row.toarray()[0][0] < 1e-06
                ):
                    break
                centroid = current_neighbors[0]
            if centroid in visited_ids:
                continue
            visited_ids.update(current_neighbors.tolist())
            pre_clusters.append(current_neighbors.tolist())
        log_time(
            message="Computing pre_clusters",
            start_time=pre_cluster_timer,
            logger=logger,
            level="debug",
        )
        pre_clusters = sorted(pre_clusters, key=len, reverse=True)
        existing_ids = set()
        clusters = []
        for cluster in pre_clusters:
            clusters.append(list(filter(lambda x: x not in existing_ids, cluster)))
            existing_ids.update(cluster)
        clusters = sorted(clusters, key=len, reverse=True)
        clusters = [cluster for cluster in clusters if len(cluster) > 0]

        log_time(
            message=f"Clustering {len(embeddings):,} embeddings to {len(clusters):,} clusters",
            start_time=timer,
            logger=logger,
        )
        return clusters
