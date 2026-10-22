from collections import defaultdict
import logging
import re
import time
from typing import List, Literal
import numpy as np
import pandas as pd
from tqdm import tqdm

from config_new_prompts import (
    DESCRIPTION_GENERATION_ITERATIONS,
    MAX_GROUP_RATIO,
    MAX_NUM_CLUSTERS,
    MAX_SIMILARITY,
    MIN_GROUP_ABSOLUTE,
    MIN_GROUP_RATIO,
    MIN_KNOWLEDGE_ATTRIBUTES,
    MIN_SIMILARITY,
    PROCESS_CHUNK_SIZE,
    STYLE_VECTOR_SIZE,
    SELECTED_STYLE_PROMPT_IDS,
    SELECTED_KNOWLEDGE_PROMPT_IDS
)
from data import prompts
from data.prompts import get_sentence_prompt#, get_knowledge_sentence_prompt
from helper import log_time
from utils.sentence_transformer import SentenceTransformerModel
from utils.llama import Llama
from utils.database import Database

logger = logging.getLogger(__name__)


def generate_style_descriptions(llama: Llama):
    database = Database()
    answers = database.get_answers_without_description(
        ["sfam_train", "sfam_val", "sfam_test"]
    )
    if len(answers) == 0:
        logger.info("All answers have style descriptions; skipping generation")
        return
    prompt_df = database.get_prompt_templates()
    input_df = pd.DataFrame()
    for _, prompt_row in prompt_df.iterrows():
        prompt_template = prompt_row["prompt_template"]
        prompts = answers["answer"].apply(lambda x: prompt_template.format(x)).tolist()
        answer_ids = [
            id
            for id in answers["id"].tolist()
            for _ in range(DESCRIPTION_GENERATION_ITERATIONS)
        ]
        input_df = pd.concat(
            [
                input_df,
                pd.DataFrame(
                    {
                        "answer_id": answer_ids,
                        "prompt_id": prompt_row["id"],
                        "prompts": prompts,
                    }
                ),
            ]
        )
    full_time = time.time()
    input_counter = 0
    for input_chunk in [
        input_df[i : i + PROCESS_CHUNK_SIZE]
        for i in range(0, len(input_df), PROCESS_CHUNK_SIZE)
    ]:
        input_counter += len(input_chunk)
        timer = time.time()
        responses = llama.generate(
            input_chunk["prompts"].to_list(),
            n_iterations=DESCRIPTION_GENERATION_ITERATIONS,
        )
        response_df = pd.DataFrame(
            {
                "answer_id": input_chunk["answer_id"].to_list(),
                "prompt_id": input_chunk["prompt_id"].to_list(),
                "style_description": [
                    response
                    for response_iterations in responses
                    for response in response_iterations
                ],
            }
        )
        remaining_time = (
            (len(input_df) - input_counter) * (time.time() - full_time) / input_counter
        )
        database.insert_style_descriptions(response_df)
        log_time(
            message=f"Generating {len(response_df):,} style descriptions for {len(input_chunk):,} inputs ({DESCRIPTION_GENERATION_ITERATIONS} iterations)",
            start_time=timer,
            logger=logger,
            remaining_time=remaining_time,
            total_time=time.time() - full_time,
        )
    database.backup("style_description")


def generate_style_sentences(llama: Llama):
    database = Database()
    descriptions = database.get_style_descriptions_with_sentenceOutputCount()
    descriptions = descriptions[descriptions["output_count"] == 0]
    if len(descriptions) == 0:
        logger.info("All style descriptions have style sentences; skipping generation")
        return

    prompt_template = get_sentence_prompt()
    knowledge_prompt_template = get_knowledge_sentence_prompt()
    
    prompt_df = database.get_prompt_templates()
    style_prompt_ids = prompt_df[(prompt_df["type"] == "Open prompt") | (prompt_df["type"] == "Targeted prompt")]["id"]
    knowledge_prompt_ids = prompt_df[(prompt_df["type"] == "Knowledge prompt")]["id"]

    def select_template(row):
        if row["prompt_id"] in style_prompt_ids:
            # Style-related descriptions
            return prompt_template.format(row["style_description"])
        if row["prompt_id"] in knowledge_prompt_ids:
            # Knowledge-related descriptions
            return knowledge_prompt_template.format(row["style_description"])
    
    full_timer = time.time()
    description_counter = 0
    for description_chunk in [
        descriptions[i : i + PROCESS_CHUNK_SIZE]
        for i in range(0, len(descriptions), PROCESS_CHUNK_SIZE)
    ]:
        timer = time.time()
        description_counter += len(description_chunk)
        prompts = description_chunk.apply(select_template, axis=1).tolist()
        
        description_ids = description_chunk["id"].tolist()
        responses = llama.generate(prompts)
        response_df = pd.DataFrame(
            {
                "description_id": description_ids,
                "output": [
                    response
                    for response_iterations in responses
                    for response in response_iterations
                ],
            }
        )
        database.insert_style_sentence_outputs(response_df)
        remaining_time = (
            (len(descriptions) - description_counter)
            * (time.time() - full_timer)
            / description_counter
        )
        log_time(
            message=f"Generating {len(response_df):,} style sentence outputs for {len(prompts):,} descriptions",
            start_time=timer,
            logger=logger,
            remaining_time=remaining_time,
            total_time=time.time() - full_timer,
        )
    database.backup("style_sentence")


def split_style_sentences(llm_enabled=True):
    database = Database()
    regex = re.compile(r"^(\d*\W+)(?=the author)", re.IGNORECASE)
    embedding_model = SentenceTransformerModel(llm_enabled=llm_enabled)
    timer = time.time()
    response_df = database.get_style_sentence_outputs_missing_sentences()
    if response_df.empty:
        logger.info("All style sentence outputs have been split; skipping splitting")
        return
    sentence_df_list = []
    skipped_sentences = 0
    regexes = {
        "negations": re.compile(r" not "),
        "avoid": re.compile(r" avoid[s]? "),
        # not necessary
        # "mention": re.compile(r" mention[s]? "),
        # not necessary
        # "quotes": re.compile(r"\""), # single quotes are fine, mostly used for e.g. author's
    }
    removed_sentences = defaultdict()
    for _, response_row in tqdm(
        response_df.iterrows(),
        total=len(response_df),
        desc="Splitting style sentences",
        smoothing=0,
    ):
        raw_sentences = embedding_model.tokenize(response_row["output"])
        raw_sentences = [
            re.sub(regex, "", sentence.strip()) for sentence in raw_sentences
        ]
        sentences = list(
            set(
                sentence.strip()
                for sentence in raw_sentences
                if sentence.strip().lower().startswith("the author ")
                or sentence.strip().lower().startswith("the author's ")
            )
        )
        if len(sentences) == 0:
            skipped_sentences += 1
            continue

        # remove unwanted sentences
        for key, regex in regexes.items():
            old_length = len(sentences)
            sentences = [
                sentence for sentence in sentences if not regex.search(sentence)
            ]
            removed_sentences[key] = (
                removed_sentences.get(key, 0) + old_length - len(sentences)
            )

        embeddings = embedding_model.encode(sentences)
        sentence_df_list.append(
            pd.DataFrame(
                {
                    "description_id": [response_row["description_id"]] * len(sentences),
                    "style_sentence": list(sentences),
                    "embedding": [embeddings[i] for i in range(len(embeddings))],
                }
            )
        )
    sentence_df = pd.concat(sentence_df_list, ignore_index=True)
    if skipped_sentences > 0:
        logger.warning(
            f"Skipped {skipped_sentences:,} sentence outputs which did not include 'the author'"
        )
    for key, value in removed_sentences.items():
        logger.warning(f"Removed {value:,} (not unique) sentences containing '{key}'")
    log_time(
        message=f"Splitting {len(sentence_df):,} style sentences from {len(response_df):,} outputs",
        start_time=timer,
        logger=logger,
    )
    database.insert_multiple_style_sentences(sentence_df)
    database.processed_sentence_outputs(response_df["id"].tolist())


def compute_clusters(recompute=False):
    database = Database()
    if recompute:
        database.delete_clusters()
    if database.count_style_sentences_missing_cluster() == 0:
        logger.info("No style sentences missing cluster; skipping clustering")
        return
    _compute_clusters(
        prompt_types=["Open prompt", "Targeted prompt"],
        cluster_prompt_type="Style prompt",
    )
    _compute_clusters(
        prompt_types=["Knowledge prompt"], cluster_prompt_type="Knowledge prompt"
    )


def _compute_clusters(
    cluster_prompt_type: Literal["Knowledge prompt", "Style prompt"],
    prompt_types: List[
        Literal["Open prompt", "Targeted prompt", "Knowledge prompt"]
    ] = [],
):
    database = Database()
    logger.debug(f"Started clustering for {prompt_types}")
    style_sentences = database.get_style_sentences_and_occurences(
        prompt_type=prompt_types
    )
    style_sentences.reset_index(drop=True, inplace=True)
    embeddings = np.array(style_sentences["embedding"].values.tolist())
    clusters = SentenceTransformerModel.cluster(embeddings, threshold=MIN_SIMILARITY)
    clusters = [
        [int(style_sentences.iloc[item]["id"]) for item in cluster]
        for cluster in clusters
    ]
    database.insert_clusters(
        [cluster[0] for cluster in clusters], prompt_type=cluster_prompt_type
    )
    logger.debug("Inserted clusters")

    cluster_df = pd.read_sql("SELECT * FROM clusters", database._conn)
    style_sentences["cluster_id"] = -1
    for cluster in clusters:
        cluster_center_id = cluster[0]
        cluster_id = int(
            cluster_df["id"][cluster_df["center_id"] == cluster_center_id].values[0]
        )
        style_sentences.loc[style_sentences["id"].isin(cluster), "cluster_id"] = (
            cluster_id
        )
    database.update_style_sentence_clusters(style_sentences[["id", "cluster_id"]])
    logger.debug("Updated style sentences with cluster ids")


def select_clusters(recompute=False):
    if recompute:
        database = Database()
        database.unaccept_clusters()
    _select_targeted()
    _pre_select_1()
    _pre_select_2()
    _final_select()


def _select_targeted(debug=True):
    database = Database()
    if database.count_accepted_clusters(status="targeted") > 0:
        logger.info(
            "There are already target clusters; skipping target cluster selection"
        )
        return
    model = SentenceTransformerModel()
    timer = time.time()

    target_prompts = prompts.get_description_prompts()
    target_prompts = target_prompts[target_prompts["type"] == "Targeted prompt"]
    target_sentences = pd.DataFrame()
    target_sentences["name"] = target_prompts["name"]
    target_sentences["sentence"] = target_prompts["name"].apply(
        lambda x: f"The author uses {x}"
    )
    target_sentences["embedding"] = model.encode(
        target_sentences["sentence"].tolist()
    ).tolist()
    target_sentences["embedding"] = target_sentences["embedding"].apply(
        lambda x: np.array(x, dtype=np.float32)
    )
    target_sentences.reset_index(drop=True, inplace=True)

    clusters = database.get_clusters(cluster_type="style")
    cluster_embeddings = np.array(clusters["embedding"].tolist(), dtype=np.float32)
    target_embeddings = np.array(
        target_sentences["embedding"].tolist(), dtype=np.float32
    )
    distances = model.similarities(target_embeddings, cluster_embeddings)
    targeted_clusters = pd.DataFrame()
    for i, row in target_sentences.iterrows():
        max_index = np.argmax(distances[i])
        targeted_clusters = pd.concat(
            [
                targeted_clusters,
                pd.DataFrame(
                    {
                        "id": [clusters.iloc[max_index]["id"]],
                        "name": [row["name"]],
                        "style_sentence": [clusters.iloc[max_index]["style_sentence"]],
                        "style_vector_index": i,
                    }
                ),
            ],
            ignore_index=True,
        )
        if debug:
            print(
                f"{row['name']}: {clusters.iloc[max_index]['style_sentence']} - {distances[i][max_index]}"
            )
    log_time(
        f"Selecting {len(targeted_clusters)} clusters as target clusters",
        start_time=timer,
        logger=logger,
    )
    database.accept_target_clusters(targeted_clusters)


def _pre_select_1():
    database = Database()
    if (
        database.count_accepted_clusters(status="pre_accepted1", include_later=False)
        > 0
    ):
        logger.info(
            "There are already pre_accepted1 clusters; skipping pre selection 1 cluster selection"
        )
        return
    min_answers = 0
    num_groups = database.count_groups(["sfam_train", "sfam_val", "sfam_test"])
    min_groups = max(MIN_GROUP_ABSOLUTE, round(num_groups * MIN_GROUP_RATIO))
    max_groups = round(num_groups * MAX_GROUP_RATIO)
    clusters = database.get_clusters_with_answers_groups(
        "rejected", include_later=False
    )
    selected_clusters = pd.DataFrame()
    first = True
    while len(selected_clusters) > MAX_NUM_CLUSTERS or first:
        first = False
        min_answers += 1
        selected_clusters = clusters[
            (clusters["answer_count"] >= min_answers)
            & (clusters["group_count"] >= min_groups)
            & (clusters["group_count"] <= max_groups)
        ]
    logger.info(
        f"Selected {len(selected_clusters):,} clusters with at least {min_answers} answers, at least {min_groups} groups and at most {max_groups} groups for pre select 1"
    )
    database.pre_accept1_clusters(
        min_groups=min_groups, max_groups=max_groups, min_answers=min_answers
    )


def _pre_select_2(debug=True):
    """Reject clusters that are too similar to each other.

    The clusters are selected in order of their occurance; each cluster that is too similar to a previous cluster is rejected.
    """
    database = Database()
    if (
        database.count_accepted_clusters(status="pre_accepted2", include_later=False)
        > 0
    ):
        logger.info(
            "There are already pre_accepted2 clusters; skipping pre selection 2 cluster selection"
        )
        return
    target_clusters = database.get_clusters(status="targeted")
    clusters = database.get_clusters(status="pre_accepted1", include_later=False)
    clusters.sort_values("occurance", ascending=False, inplace=True)
    clusters = pd.concat([target_clusters, clusters], ignore_index=True)
    similarities = SentenceTransformerModel.similarities(
        clusters["embedding"].values.tolist()
    )
    flat_similarities = np.triu(similarities, k=1).flatten()
    delete_similar = np.argwhere(flat_similarities >= MAX_SIMILARITY)
    current_index = 0
    delete_clusters = set()
    targeted_ids = set(target_clusters.index)
    first = True
    for i in range(len(clusters)):
        skip_index = False
        if i in delete_clusters:
            skip_index = True
        this_index = current_index
        while delete_similar[this_index] < (i + 1) * len(clusters):
            this_index += 1
            if this_index >= len(delete_similar):
                break
        else:
            if this_index > current_index and not skip_index:
                delete_ids = set(
                    delete_similar[current_index:this_index].flatten()
                    - i * len(clusters)
                )
                if debug and len(targeted_ids.intersection(delete_ids)) > 0:
                    if first:
                        print(
                            f"The following target clusters have a cosine similarity >= {MAX_SIMILARITY} to another target cluster:"
                        )
                        first = False
                    print(f"{i:02}: {clusters.iloc[i]['style_sentence']}")
                    for j in targeted_ids.intersection(delete_ids):
                        print(
                            f"    - {j:02}: {clusters.iloc[j]['style_sentence']} ({flat_similarities[j + i * len(clusters)]})",
                        )
                delete_clusters.update(delete_ids)
            current_index = this_index
            continue
        break
    targeted_delete = delete_clusters.intersection(targeted_ids)
    if len(targeted_delete) != 0:
        logger.warning(
            f"{len(targeted_delete)} target clusters would have been selected to be deleted (ids {','.join([str(i) for i in sorted(targeted_delete)])})"
        )
    pre_accepted2_clusters = clusters.drop(delete_clusters)
    logger.info(f"Selected {len(pre_accepted2_clusters):,} clusters for pre select 2")
    pre_accepted2_clusters = pre_accepted2_clusters[
        ~pre_accepted2_clusters["id"].isin(target_clusters["id"])
    ]
    database.update_cluster_status(
        pre_accepted2_clusters["id"].values.tolist(), "pre_accepted2"
    )


def _final_select():
    database = Database()
    if database.count_accepted_clusters(status="accepted") > 0:
        logger.info(
            "There are already accepted clusters; skipping final cluster selection"
        )
        return
    clusters = database.get_clusters(status="pre_accepted2", include_later=True)
    clusters.sort_values("id", inplace=True, ascending=True)
    similarities = SentenceTransformerModel.similarities(
        clusters["embedding"].values.tolist()
    )
    answers = database.get_answers_with_clusterId_list(
        ["sfam_train", "sfam_val", "sfam_test"], "pre_accepted2"
    )
    cluster_indeces = answers["cluster_id_list"].values
    averages = []
    for index in cluster_indeces:
        sims = similarities[index]
        averages.append(np.mean(sims, axis=0))
    averages_df = pd.DataFrame(
        {
            "answer_id": answers["answer_id"].tolist(),
            "cluster_similarities": averages,
            "cluster_data_points": answers["cluster_id_data"].tolist(),
        }
    )
    result = pd.DataFrame()
    for index, cluster_id in enumerate(clusters["id"].values):
        current_answer = averages_df.copy()
        current_answer["similarity"] = current_answer["cluster_similarities"].apply(
            lambda x: x[index]
        )
        current_answer["data_point"] = current_answer["cluster_data_points"].apply(
            lambda x: x[index]
        )
        current_answer.sort_values("similarity", ascending=False, inplace=True)
        positives = current_answer[current_answer["data_point"] == 1]
        negatives = current_answer[current_answer["data_point"] == 0]
        result = pd.concat(
            [
                result,
                pd.DataFrame(
                    [
                        {
                            "cluster_id": cluster_id,
                            "prompt_type": clusters[clusters["id"] == cluster_id][
                                "prompt_type"
                            ].values[0],
                            "pos_sim": positives["similarity"].mean(),
                            "neg_sim": negatives["similarity"]
                            .tail(len(positives))
                            .mean(),
                            "range": positives["similarity"].mean()
                            - negatives["similarity"].tail(len(positives)).mean(),
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
    result.sort_values("range", ascending=False, inplace=True)
    target_clusters = database.get_clusters(status="targeted")
    # merge to preserve the order in result
    selected_clusters = result.head(STYLE_VECTOR_SIZE)[["cluster_id"]].merge(
        clusters, how="left", left_on="cluster_id", right_on="id"
    )
    # remove target clusters as they are already selected; the cluster status is updated at the end
    selected_clusters = selected_clusters[
        ~selected_clusters["id"].isin(target_clusters["id"])
    ].head(STYLE_VECTOR_SIZE - len(target_clusters))

    # make sure that at least MIN_KNOWLEDGE_ATTRIBUTES clusters with prompt_type "Knowledge prompt" are included
    selected_knowledge_clusters = selected_clusters[
        selected_clusters["prompt_type"] == "Knowledge prompt"
    ]
    if len(selected_knowledge_clusters) < MIN_KNOWLEDGE_ATTRIBUTES:
        knowledge_cluster_results = result[result["prompt_type"] == "Knowledge prompt"]
        selected_knowledge_cluster_ids = knowledge_cluster_results.head(
            MIN_KNOWLEDGE_ATTRIBUTES
        )[["cluster_id"]]
        selected_knowledge_clusters = selected_knowledge_cluster_ids.merge(
            clusters, how="left", left_on="cluster_id", right_on="id"
        )
        selected_clusters = selected_clusters[
            selected_clusters["prompt_type"] != "Knowledge prompt"
        ]
        selected_clusters = selected_clusters.head(
            STYLE_VECTOR_SIZE
            - len(target_clusters)
            - len(selected_knowledge_cluster_ids)
        )
        selected_clusters = pd.concat(
            [
                selected_knowledge_clusters,
                selected_clusters,
            ],
            ignore_index=True,
        )
    # sort the attributes, so that in the style vector it is "targeted", "knowledge", then "style"
    selected_clusters = selected_clusters.sort_values(
        "prompt_type", ascending=True
    ).reset_index(drop=True)

    selected_clusters["style_vector_index"] = range(
        len(target_clusters), STYLE_VECTOR_SIZE
    )
    logger.info(
        f"Selected {len(selected_clusters) + len(target_clusters):,} clusters for final selection"
    )
    database.accept_final_clusters(selected_clusters)
    database.update_cluster_status(target_clusters["id"].values.tolist(), "accepted")


def create_data_style_vectors():
    database = Database()
    answer_count = database.count_answers_without_stylevector(
        vector_type="data",
        data_types=["sfam_train", "sfam_val", "sfam_test"],
    )
    if answer_count == 0:
        logger.info(
            "All answers have data style vectors; skipping style vector generation"
        )
        return
    answers = database.get_answers_with_styleVectorIndex(
        ["sfam_train", "sfam_val", "sfam_test"]
    )
    database.update_answer_styleVector(data=answers, vector_type="data")
    logger.info(f"Created {len(answers):,} data style vectors")


def create_similarity_style_vectors():
    database = Database()
    answer_count = database.count_answers_without_stylevector(
        vector_type="similarity",
        data_types=["sfam_train", "sfam_val", "sfam_test"],
    )
    if answer_count == 0:
        logger.info(
            "All answers have similarity style vectors; skipping style vector generation"
        )
        return
    clusters = database.get_clusters(status="accepted")
    clusters.sort_values("style_vector_index", inplace=True, ascending=True)
    similarities = SentenceTransformerModel.similarities(
        clusters["embedding"].values.tolist()
    )
    answers = database.get_answers_with_styleVectorIndex(
        ["sfam_train", "sfam_val", "sfam_test"]
    )
    style_vector_indeces = answers["style_vector_index"].values
    averages = []
    for index in style_vector_indeces:
        sims = similarities[index]
        averages.append(np.nan_to_num(np.mean(sims, axis=0), nan=0.0))
    # create dataframe with answer_id from answers and style_vector from averages
    averages_df = pd.DataFrame(
        {
            "answer_id": answers["answer_id"].tolist(),
            "style_vector": averages,
        }
    )
    database.update_answer_styleVector(data=averages_df, vector_type="similarity")
    logger.info(f"Created {len(answers):,} similarity style vectors")
    database.backup("base_vector")


def create_sfam_style_vectors():
    pass


def create_lisa_style_vectors():
    pass
