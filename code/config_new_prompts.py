from typing import List, Literal, Union

LOGGING_FOLDER = "logs/"

# Model to extract the attributes
GENERATION_MODEL = "/path/to/folder/Qwen2.5-7B-Instruct" #Llama-3.2-3B-Instruct"
BACKUP_FOLDER = "backup/"
MODEL_OUTPUT_FOLDER = "models/"
RESULT_FOLDER = "results/"

STEERING_MODEL = ["/path/to/folder/Qwen2.5-7B-Instruct", "/path/to/folder/Llama-3.1-8B-Instruct", "/path/to/folder/Ministral-8B-Instruct-2410",  "/path/to/folder/Qwen2.5-72B-Instruct"]
SENTENCE_TRANSFORMER_MODEL = "/path/to/folder/Qwen3-Embedding-8B"
SFAM_MODEL = "/path/to/folder/deberta-v3-large"
LISA_MODEL = "/path/to/folder/deberta-v3-large"
# when sfam has a lower accuracy on the validation set, the program stops
SFAM_MIN_ACCURACY = 0.85

STYLE_VECTOR_SIZE = 5000
STYLE_EMBEDDING_SIZE = 10000

# how many knowledge clusters are dimensions of the attribute vector
# if there are less candidates, the actual number can be lower
# if more are selected anyways, the number can be higher
MIN_KNOWLEDGE_ATTRIBUTES = None

DESCRIPTION_GENERATION_ITERATIONS = 1
# Number of prompts that are batch generated; after each batch the result is saved
PROCESS_CHUNK_SIZE = 10_000
# The maximum number of clusters that get excepted for further processing
MAX_NUM_CLUSTERS = 10_000

# the maximum ratio of groups a cluster is used to describe for it to be selected (too many groups mean it is not usefull for differentiation)
MAX_GROUP_RATIO = 0.6
# the minimum number of groups is the maximum of the ratio and the absolute
## the minimum ratio of groups a cluster is used to describe for it to be selected (minimum of 0)
MIN_GROUP_RATIO = 0
## the minimum number of groups a cluster is used to describe for it to be selected (minimum of 1)
MIN_GROUP_ABSOLUTE = 1

# the minimum similarity two sentences must have to be clustered together
MIN_SIMILARITY = 0.85
# the maximum similarity that two clusters can have to be selected as a style attribute
MAX_SIMILARITY = 0.7

# how many samples are selected for each group
# the data types have to be one of helper.DATA_TYPE_ENUM
# the sfam data types will be used as lisa_train as well
DATASET_MAPPING = {
    "stackex": {
        "sfam_train": 400,
        "sfam_val": 50,
        "sfam_test": 50,
        "lisa_train": 5000,
        "lisa_val": 300,
        "lisa_test": 200,
    },
    "askx": {
        "foreign_domain_test": 500,
    },
}
# if the dataset mapping mode is group, the keys of the dataset_mapping dict are the group names
# if the dataset mapping mode is dataset, the keys of the dataset_mapping dict are the dataset names from which the data is imported; the values are applied to all groups in the dataset
DATASET_MAPPING_MODE: Literal["dataset", "group"] = "dataset"
MULTIPLE_ANSWERS_PER_QUESTION = True

# the training of the models can be optimized with hyperparameter optimization
# this variable defines the name of the sqlite file that is used to store the results
HPO_FILE = "hyperparameter_optimization.sqlite"

# rename the group names of groups that are used for steering to better names for the steering system prompt
GROUP_STEERING_NAMES = {
    "astronomy": "astronomers",
    "aviation": "pilots",
    "biology": "biologists",
    "chemistry": "chemists",
    "crypto": "cryptographers",
    "cs": "computer scientists",
    "datascience": "data scientists",
    "electronics": "electrical engineers",
    "gamedev": "game developers",
    "history": "historians",
    "law": "lawyers",
    "mechanics": "mechanical engineers",
    "philosophy": "philosophers",
    "physics": "physicists",
    "politics": "politicians",
    "softwareengineering": "software engineers",
}

# this should be style_vector_lisa, but in case sfam should be used as the basis for embedding and steering use style_vector_sfam
# Vector basis for which the most important attributes are calculated
STYLE_EMBEDDING_VECTOR = "style_vector_data"

# how many attributes will be used for steering
NUM_IMPORTANT_ATTRIBUTES_STYLE = None
NUM_IMPORTANT_ATTRIBUTES_KNOWLEDGE = None
NUM_IMPORTANT_ATTRIBUTES = [3, 5, 10]
# if false, all dimensions of the attribute vector are used for activation steering
# if true, only the most important `NUM_IMPORTANT_ATTRIBUTES_STYLE` + `NUM_IMPORTANT_ATTRIBUTES_KNOWLEDGE` dimensions are used for activation steering
USE_IMPORTANT_ATTRIBUTES_FOR_ACTIVATION_STEERING = True
# the steering vecor is multiplied with the lambda to change the steering strength
# the activation steering is repeated for each lambda
ACTIVATION_SPACE_LAMBDAS = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
KONEN_ACTIVATION_SPACE_LAMBDAS = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5]
# use these layers for activation steering (either one or multiple at once)
# the activation steering is repeated for each entry in the list
ACTIVATION_STEERING_INSERT_LAYER_INDECES = {"/path/to/folder/Qwen2.5-7B-Instruct" : [list(range(start, start + 3)) for start in range(0, 28 - 3 + 1)],
                                            "/path/to/folder/Llama-3.1-8B-Instruct" : [list(range(start, start + 3)) for start in range(0, 32 - 3 + 1)],
                                            "/path/to/folder/Ministral-8B-Instruct-2410" : [list(range(start, start + 3)) for start in range(0, 36 - 3 + 1)],
                                           "/path/to/folder/Qwen2.5-32B-Instruct": [list(range(start, start + 3)) for start in range(24, 54 - 3 + 1)]}
print(ACTIVATION_STEERING_INSERT_LAYER_INDECES)

# the evaluation results are printed in results.md or results.tex
RESULT_FILE_FORMAT: Literal["markdown", "latex"] = "markdown"

SELECTED_STYLE_PROMPT_IDS = list(range(1, 93)) #[1, ..., 92] #[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 22, 23, 24, 25, 27, 28, 29, 30, 31, 32, 33, 34, 64, 65, 66, 67, 68, 69, 70, 71, 72, 74, 75, 85, 86, 87, 88, 89, 90, 91, 92]  
#


SELECTED_KNOWLEDGE_PROMPT_IDS = [93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110]
