import argparse
import logging
import time

from helper import log_time
from scripts import embedder, lisa, steering, sfam, normal_embedder
from scripts.debugging import (
    debug_generation,
    debug_repeating_outputs,
    debug_similar_sentences,
    debug_skipped_sentence_outputs,
)
from scripts.evaluate import evaluate_all
from scripts.sentence_generation import (
    compute_clusters,
    create_data_style_vectors,
    create_similarity_style_vectors,
    generate_style_descriptions,
    generate_style_sentences,
    select_clusters,
    split_style_sentences,
)
from utils.logger import setup_logging
from config_new_prompts import GENERATION_MODEL, LOGGING_FOLDER, STEERING_MODEL, NUM_IMPORTANT_ATTRIBUTES
from utils.llama import Llama
from utils.database import Database

logger = logging.getLogger(__name__)

DATABASE_NAME = "data.sqlite"


def main():
    # import data

    # data.convert_stackex_data()
    # data.convert_askx_data()
    database = Database(DATABASE_NAME)
    database.create_database()
    if database.new_db:
        # all cleanup that can not be done in the database class
        # due to circular dependencies
        sfam.delete_model()
        lisa.delete_model()
        embedder.delete_model()
    llama = Llama(
        model_id=GENERATION_MODEL,
        enabled=args.llm_enabled,
    )

    generate_style_descriptions(llama=llama)
    generate_style_sentences(llama=llama)
    del llama

    split_style_sentences()
    compute_clusters()
    select_clusters()

    create_data_style_vectors()
    create_similarity_style_vectors()

    if args.llm_enabled:
        sfam.train_and_create_style_vector(
            recompute=False, do_hyperparameter_optimization=False
        )
        lisa.train_and_create_style_vector(
            recompute=False, do_hyperparameter_optimization=False
        )
        embedder.train_and_create_style_embedding(
            recompute=False, do_hyperparameter_optimization=False
        )
        normal_embedder.create_normal_embedding()
        
        for steering_model in STEERING_MODEL:
            for num_important_attributes in NUM_IMPORTANT_ATTRIBUTES:
                steering.steer(steering_model, num_important_attributes, recompute=False)

    evaluate_all(do_evaluate_sfam=args.llm_enabled)

    return



if __name__ == "__main__":
    timer = time.time()
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--llm_enabled",
        help="Enable llama and other ML models",
        action="store_true",
    )
    argparser.add_argument(
        "--continuation",
        help="Continuation of the program (will append to the existing log)",
        action="store_true",
    )
    args = argparser.parse_args()
    setup_logging(
        continuation=args.continuation,
        folder=LOGGING_FOLDER,
        filtered_loggers=["urllib3.connectionpool", "filelock"],
    )
    logger.info("Starting the program")
    main()
    logger.info("Program finished")
    log_time("The program", start_time=timer, logger=logger)
