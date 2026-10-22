from collections import defaultdict
import logging
from pathlib import Path
import random
from typing import Dict, List, Literal, Tuple, Union
import numpy as np
import pandas as pd

from utils.sentence_transformer import SentenceTransformerModel
from utils.database import Database

logger = logging.getLogger(__name__)


def debug_generation(folder: str, n_outputs=10, shorten=-1):
    database = Database()
    if n_outputs <= 0:
        logger.info("Skipping debug generation as n_outputs is less than 1")
    folder = folder + "/random_samples"
    Path.mkdir(Path(folder), exist_ok=True, parents=True)

    cursor = database._conn.cursor()
    cursor.execute(
        """
        SELECT answers.id
        FROM answers
        JOIN style_descriptions ON answers.id = style_descriptions.answer_id
        GROUP BY answers.id
        HAVING COUNT(style_descriptions.id) > 0
        """
    )
    output_ids = cursor.fetchall()
    random.seed(42)
    if len(output_ids) > n_outputs:
        output_ids = random.sample(output_ids, n_outputs)
    for output_id in output_ids:
        output_id = output_id[0]
        output = pd.read_sql_query(
            "SELECT * FROM answers WHERE id = ?",
            database._conn,
            params=(int(output_id),),
        )
        # get the group of the output
        cursor.execute(
            """
            SELECT questions.data_group
            FROM answers
            JOIN questions ON answers.question_id = questions.id
            WHERE answers.id = ?
            """,
            (output_id,),
        )
        group = cursor.fetchone()[0]
        with open(
            f"{folder}/output_{output_id:05}_{group}.txt",
            "w",
            encoding="utf-8",
        ) as f:
            f.write(output["answer"].values[0])
            f.write("\n\n" + "=" * 80 + "\n\n")
            style_descriptions = database.get_style_descriptions_by_answerId(output_id)
            for _, row in style_descriptions.iterrows():
                style_description = row["style_description"]
                if shorten > 0:
                    style_description = style_description[:shorten] + "..."
                f.write(style_description)
                f.write("\n---\n")
                cursor.execute(
                    """
                    SELECT output
                    FROM style_sentence_outputs
                    WHERE description_id = ?
                    """,
                    (row["id"],),
                )
                style_sentences_output = cursor.fetchone()[0]
                if shorten > 0:
                    style_sentences_output = style_sentences_output[:shorten] + "..."
                f.write(style_sentences_output)
                style_sentences = database.get_style_sentences(row["id"])
                f.write("\n" + "-" * 40 + "\n")
                for _, row in style_sentences.iterrows():
                    style_sentence = row["style_sentence"]
                    f.write(style_sentence + "\n")
                f.write("=" * 40 + "\n")


def debug_repeating_outputs(folder: str, n_outputs=10, n_gram=10, min_repitions=20):
    import nltk  # type: ignore

    database = Database()
    if n_outputs <= 0:
        logger.info("Skipping debug generation as n_outputs is less than 1")

    description_folder = folder + "/repeating_descriptions"
    sentence_folder = folder + "/repeating_sentence_outputs"
    Path.mkdir(Path(description_folder), exist_ok=True, parents=True)
    Path.mkdir(Path(sentence_folder), exist_ok=True, parents=True)

    outputs = pd.read_sql_query(
        """
        SELECT style_descriptions.id, style_descriptions.style_description, style_sentence_outputs.output, prompts.type, prompts.name
        FROM style_descriptions
        JOIN prompts ON style_descriptions.prompt_id = prompts.id
        JOIN style_sentence_outputs ON style_descriptions.id = style_sentence_outputs.description_id
        """,
        database._conn,
    )
    outputs["description_length"] = outputs["style_description"].apply(lambda x: len(x))
    outputs["sentence_output_length"] = outputs["output"].apply(lambda x: len(x))
    outputs[f"description_repeating_{n_gram}_grams"] = outputs[
        "style_description"
    ].apply(
        lambda x: len(
            [
                ngram
                for ngram in nltk.ngrams(x.split(), n_gram)
                if x.count(" ".join(ngram)) > 1
            ]
        )
    )
    outputs[f"sentence_repeating_{n_gram}_grams"] = outputs["output"].apply(
        lambda x: len(
            [
                ngram
                for ngram in nltk.ngrams(x.split(), n_gram)
                if x.count(" ".join(ngram)) > 1
            ]
        )
    )
    repeating_descriptions = outputs[
        outputs[f"description_repeating_{n_gram}_grams"] > min_repitions
    ]
    if not repeating_descriptions.empty:
        logger.info(
            f"Number of style descriptions with repeating {n_gram}-grams: {len(repeating_descriptions):,} of {len(outputs):,} ({len(repeating_descriptions) / len(outputs) * 100:.2f}%); the description with repitions have an average of {repeating_descriptions[f'description_repeating_{n_gram}_grams'].mean():,.1f} repititions per description"
        )
    repeating_descriptions.sort_values(
        f"description_repeating_{n_gram}_grams", ascending=False, inplace=True
    )
    repeating_descriptions.reset_index(drop=True, inplace=True)
    for index, row in repeating_descriptions[:n_outputs].iterrows():
        with open(
            f"{description_folder}/description_{index + 1:02}_{row['id']:05}.txt",
            "w",
            encoding="utf-8",
        ) as f:
            f.write(f"Prompt: {row['type']} - {row['name']}\n" + "=" * 80 + "\n\n")
            f.write(row["style_description"])

    repeating_sentence_outputs = outputs[
        outputs[f"sentence_repeating_{n_gram}_grams"] > min_repitions
    ]
    if not repeating_sentence_outputs.empty:
        logger.info(
            f"Number of sentence outputs with repeating {n_gram}-grams: {len(repeating_sentence_outputs):,} of {len(outputs):,} ({len(repeating_sentence_outputs) / len(outputs) * 100:.2f}%); "
            + f"the sentence with repitions have an average of {repeating_sentence_outputs[f'sentence_repeating_{n_gram}_grams'].mean():,.1f} repititions per sentence"
        )
    repeating_sentence_outputs.sort_values(
        f"sentence_repeating_{n_gram}_grams", ascending=False, inplace=True
    )
    repeating_sentence_outputs.reset_index(drop=True, inplace=True)
    for index, row in repeating_sentence_outputs[:n_outputs].iterrows():
        with open(
            f"{sentence_folder}/sentence_{index + 1:02}_{row['id']:05}.txt",
            "w",
            encoding="utf-8",
        ) as f:
            f.write(f"Prompt: {row['type']} - {row['name']}\n")
            if row[f"description_repeating_{n_gram}_grams"] > min_repitions:
                f.write(
                    f"The description is repeating with {row[f'description_repeating_{n_gram}_grams']} {n_gram}-grams\n"
                )
            f.write("=" * 80 + "\n\n")
            f.write(row["output"])

    # worst prompts
    worst_description_prompts = (
        repeating_descriptions.groupby(["type", "name"])
        .size()
        .reset_index(name="count")
    )
    worst_description_prompts.sort_values("count", ascending=False, inplace=True)
    worst_description_prompts = worst_description_prompts[:n_outputs]
    worst_sentence_prompts = (
        repeating_sentence_outputs.groupby(["type", "name"])
        .size()
        .reset_index(name="count")
    )
    worst_sentence_prompts.sort_values("count", ascending=False, inplace=True)
    worst_sentence_prompts = worst_sentence_prompts[:n_outputs]
    if not worst_description_prompts.empty:
        logger.info(
            "Worst description prompts: "
            + ", ".join(
                worst_description_prompts[["name", "count"]]
                .apply(lambda x: f"{x['name']} ({x['count']})", axis=1)
                .values.tolist()
            )
        )
    if not worst_sentence_prompts.empty:
        logger.info(
            "Worst sentence prompts: "
            + ", ".join(
                worst_sentence_prompts[["name", "count"]]
                .apply(lambda x: f"{x['name']} ({x['count']})", axis=1)
                .values.tolist()
            )
        )
    # print(worst_description_prompts)
    # print(worst_sentence_prompts)


def debug_similar_sentences(
    cluster_status: Literal[
        "all", "rejected", "accepted", "pre_accepted1", "pre_accepted2"
    ] = "all",
):
    database = Database()

    def cluster_similarity(
        cluster_status: Literal[
            "all", "rejected", "accepted", "pre_accepted1", "pre_accepted2"
        ] = "all",
        min_similarity: Union[float, None] = None,
        max_similarity: Union[float, None] = None,
        k=1,
    ) -> Dict[str, List[Tuple[str, int]]]:
        all_clusters = database.get_clusters(status=cluster_status)
        similarities = SentenceTransformerModel.similarities(
            all_clusters["embedding"].values.tolist()
        )
        similar_clusters: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
        upper_tri_indices = np.triu_indices(len(similarities), k=k)
        for i, j in zip(upper_tri_indices[0], upper_tri_indices[1]):
            if min_similarity is not None and similarities[i, j] > min_similarity:
                similar_clusters[all_clusters.iloc[i]["id"]].append(
                    (all_clusters.iloc[j]["id"], similarities[i, j])
                )
                if i != j:
                    similar_clusters[all_clusters.iloc[j]["id"]].append(
                        (all_clusters.iloc[i]["id"], similarities[i, j])
                    )
            if max_similarity is not None and similarities[i, j] < max_similarity:
                similar_clusters[all_clusters.iloc[i]["id"]].append(
                    (all_clusters.iloc[j]["id"], similarities[i, j])
                )
                if i != j:
                    similar_clusters[all_clusters.iloc[j]["id"]].append(
                        (all_clusters.iloc[i]["id"], similarities[i, j])
                    )
        return similar_clusters

    def similar_clusters_df(
        cluster_similarity: Dict[str, List[Tuple[str, int]]],
        column_name="similar_count",
    ) -> pd.DataFrame:
        similar_df = database.get_clusters()[
            ["id", "size", "occurance", "style_sentence"]
        ]
        similar_df[column_name] = similar_df["id"].apply(
            lambda x: len(cluster_similarity[x]) if x in cluster_similarity else 0
        )
        similar_df.sort_values(by=column_name, ascending=False, inplace=True)
        similar_df.rename(columns={"id": "cluster_id"}, inplace=True)
        similar_df = similar_df[
            ["cluster_id", column_name, "size", "occurance", "style_sentence"]
        ]
        return similar_df

    if database.get_clusters(status=cluster_status).empty:
        logger.warning("No clusters found; skipping similarity check")
        return
    most_similars = cluster_similarity(
        min_similarity=0.85, k=1, cluster_status=cluster_status
    )
    most_similar_df = similar_clusters_df(most_similars)
    for i in range(1, most_similar_df["similar_count"].max() + 1):
        logger.warning(
            f"Number of clusters with {i} similar cluster(s): {len(most_similar_df[most_similar_df['similar_count'] == i])}"
        )


def debug_skipped_sentence_outputs(folder: str, max_outputs=10):
    database = Database()
    folder = folder + "/skipped_sentence_outputs"
    Path.mkdir(Path(folder), exist_ok=True, parents=True)
    skipped_sentence_outputs = database.get_style_sentence_outputs_missing_sentences()
    if skipped_sentence_outputs.empty:
        return
    for _, row in skipped_sentence_outputs[:max_outputs].iterrows():
        with open(f"{folder}/output_{row['id']:05}.txt", "w", encoding="utf-8") as f:
            f.write(
                f"Prompt: {row['prompt_type']} - {row['prompt_name']}, description_id: {row['description_id']}, output_id: {row['id']}\n"
            )
            f.write("=" * 80 + "\n\n")
            f.write(row["style_description"])
            f.write("\n\n" + "=" * 80 + "\n\n")
            f.write(row["output"])
