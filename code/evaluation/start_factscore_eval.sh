#!/bin/bash

STEERING_MODEL="path/to/folder/<llm_name>/"

DATABASE="data.sqlite"

# Number of most important attributes
m=5

python3.10 get_factscore.py -db "$DATABASE" -llm "$STEERING_MODEL" -approaches prompt_group
python3.10 get_factscore.py -db "$DATABASE" -llm "$STEERING_MODEL" -approaches prompt_group_attribute_"$m"
python3.10 get_factscore.py -db "$DATABASE" -llm "$STEERING_MODEL" -approaches activation_konen_group
python3.10 get_factscore.py -db "$DATABASE" -llm "$STEERING_MODEL" -approaches activation_konen_group_attribute_"$m"
python3.10 get_factscore.py -db "$DATABASE" -llm "$STEERING_MODEL" -approaches activation_group_"$m"
python3.10 get_factscore.py -db "$DATABASE" -llm "$STEERING_MODEL" -approaches activation_group_attribute_"$m"
