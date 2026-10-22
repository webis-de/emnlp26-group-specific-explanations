from factscore.factscorer import FactScorer
from utils.database import Database
import torch
torch.use_deterministic_algorithms(True)
import numpy as np

from config_new_prompts import RESULT_FOLDER
import json
import os
import shutil
import argparse

import hashlib, time

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TRANSFORMERS_NO_TORCH_COMPILE"] = "1"


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-db",
        type=str,
        required=True,
        help="Name of the database"
    )

    parser.add_argument(
        "-llm",
        type=str,
        required=True,
        help="Path to the steered LLM model"
    )

    parser.add_argument(
        "-approaches",
        type=str,
        nargs="+",
        default=[],
        help="Approaches to evaluate"
    )

    args = parser.parse_args()
    
    hash_id = hashlib.sha256(str(time.time()).encode()).hexdigest()[:16]
    cache_dir = f"/path/to/folder/data/factscore/.cache_{hash_id}/cache"
    
    scorer = FactScorer(
        afg_model="/path/to/folder/OLMo-2-1124-7B-SFT",
        afv_model="/path/to/folder/gemma-3-4b-it",
        data_dir="/path/to/folder/data/factscore",
        model_dir="/path/to/folder/data/factscore",
        cache_dir=cache_dir
    )

    DATABASE_NAME = args.db
    database = Database(DATABASE_NAME)
    database.create_database()

    steering_model = args.llm
    steered_data = database.get_steering_generation_of_llm(
        llm_name=steering_model
    )

    test_question_ids = database.get_steering_questions(
        split="rooein_test"
    )["id"].unique()

    test_steered_data = steered_data[
        steered_data["question_id"].isin(test_question_ids)
    ]

    question2topic = {
      "How do plants reproduce?": "Plant reproduction",
      "What is the history of the universe and how have scientists pieced together this timeline?": "Cosmology",
      "How do scientists study the human brain and what have they learned about its structure and function?": "Neuroscience",
      "What is the theory of evolution?": "Evolution",
      "What are the different types of cells in the human body and what are their functions?": "Cell (biology)",
      "How do we measure the strength of earthquakes?": "Seismology",
      "What is the difference between a physical and chemical property?": "Physical property",
      "What is the role of hormones in the body and how do they regulate various bodily functions?": "Endocrine system",
      "What is the role of enzymes in digestion?": "Digestive enzyme",
      "How does DNA replication work and what are the different types of mutations that can occur?": "DNA replication",
      "How do scientists study the behavior of particles at the subatomic level using particle accelerators and detectors?": "Particle physics",
      "What is the process of osmosis?": "Osmosis",
      "What is the difference between a theory and a hypothesis?": "Scientific theory",
      "What are the basic principles of physics?": "Physics",
      "How do different types of materials conduct and store electricity and how are they used in electronic devices?": "Electrical resistivity and conductivity",
      "What are the different types of energy and how do they affect the environment?": "Energy",
      "What is the impact of air pollution on human health and what steps can we take to reduce our exposure to harmful pollutants?": "Air pollution",
      "What is the difference between a conductor and an insulator?": "Electrical conductor",
      "What is the process of meiosis?": "Meiosis",
      "What are the different types of rocks and how are they formed through geological processes?": "Rock (geology)",
      "What is the difference between renewable and non-renewable energy sources?": "Renewable energy",
      "How do geologists study earthquakes and volcanoes and what are the different types of geological hazards?": "Geological hazard",
      "How does the human brain work?": "Human brain",
      "What is the process of photosynthesis?": "Photosynthesis",
      "What are the different types of biotechnology and how are they used in fields like medicine and agriculture?": "Biotechnology",
      "What are the different types of chemical reactions and how do they influence the behavior of matter and energy?": "Chemical reaction",
      "What are the different types of cells in the human body?": "Cell type",
      "How do scientists study the composition and behavior of different types of stars in the universe?": "Astrophysics",
      "How do scientists study the behavior and interactions of different species in an ecosystem?": "Ecology",
      "How do scientists study the behavior and movements of animals in their natural habitats?": "Ethology",
      "How does the human body maintain homeostasis and what happens when this balance is disrupted?": "Homeostasis",
      "How do we use radioactive isotopes in medicine?": "Nuclear medicine",
      "What is the difference between weather and climate?": "Climate",
      "How do plants convert sunlight into energy through photosynthesis?": "Photosynthesis",
      "What are the different types of biomes and what are the unique features and characteristics of each one?": "Biome",
      "How do scientists study the composition and behavior of the different layers of the Earth’s atmosphere?": "Atmosphere of Earth",
      "What is the process of mitosis?": "Mitosis",
      "What are Newton’s laws in dynamics?": "Newton's laws of motion",
      "How do ecosystems function and what are the different components that make up these complex systems?": "Ecosystem",
      "What is the difference between a hypothesis and a theory?": "Hypothesis",
      "What is the theory of relativity?": "Theory of relativity",
      "What are the different types of biomes?": "Biome",
      "What is the difference between a virus and a bacteria?": "Virus",
      "What are the different types of energy and how are they converted from one form to another?": "Energy transformation",
      "How do scientists study the behavior of matter at the atomic and subatomic levels and what are the different tools they use?": "Atomic physics",
      "What is the difference between a solid, liquid, and gas?": "State of matter",
      "What is the impact of human activity on the environment and what steps can we take to reduce our ecological footprint?": "Environmental impact assessment",
      "How do we use sound waves to diagnose medical conditions?": "Ultrasound",
      "What is the scientific method and why is it important?": "Scientific method",
      "What is the greenhouse effect and how does it contribute to climate change?": "Greenhouse effect",
      "How do scientists study the effects of drugs on the human body and what are the different approaches to drug development?": "Pharmacology",
      "What are the different types of waves in the electromagnetic spectrum and how do we use them in everyday life?": "Electromagnetic spectrum",
      "What is the structure of the atom and how does it explain chemical reactions?": "Atom",
      "How does the process of natural selection work and how does it explain the evolution of species over time?": "Natural selection",
      "How do vaccines work and how have they impacted public health?": "Vaccination",
      "How do tectonic plates move?": "Plate tectonics",
      "How do earthquakes happen?": "Earthquake",
      "What is the greenhouse effect?": "Greenhouse effect",
      "What are the fundamental laws of physics and how do they govern the behavior of the natural world?": "Scientific law",
      "What is the difference between renewable and nonrenewable resources and how do we use them to meet our energy needs?": "Natural resource",
      "How do animals communicate with each other?": "Animal communication",
      "How does the human immune system work?": "Immune system",
      "How do airplanes fly?": "Aerodynamics",
      "How do we measure the age of the earth?": "Age of Earth",
      "How do we measure temperature?": "Temperature measurement",
      "How do scientists classify living organisms?": "Taxonomy (biology)",
      "What is the structure of an atom and how do atoms combine to form molecules?": "Molecule",
      "How does the circulatory system work?": "Circulatory system",
      "How do scientists measure and track changes in the Earth’s climate over time?": "Climate change",
      "What are the different types of waves and how do they interact with matter?": "Wave",
      "What is the relationship between electricity and magnetism and how do they interact with one another?": "Electromagnetism",
      "How do we use DNA technology to identify people?": "DNA profiling",
      "How do different types of lenses and mirrors work and how are they used in telescopes and microscopes?": "Optics",
      "What is the relationship between genetics and behavior and how do scientists study this relationship?": "Behavioural genetics",
      "How do scientists study the behavior and properties of light and what are the different types of electromagnetic radiation?": "Optics",
      "What is the role of symbiosis in the natural world and how do different organisms benefit from these relationships?": "Symbiosis",
      "How do telescopes work?": "Telescope",
      "How do neurons in the brain communicate with one another and what role do neurotransmitters play in this process?": "Neuron",
      "What is the relationship between plate tectonics and geological processes like earthquakes and volcanic eruptions?": "Plate tectonics",
      "How does sound travel?": "Sound",
      "How does DNA work?": "DNA",
      "What is the process of fermentation?": "Fermentation",
      "What are the fundamental principles of physics and how do they explain the behavior of matter and energy in the natural world?": "Physics",
      "How do scientists study the deep ocean and what have they discovered about life in these extreme environments?": "Oceanography",
      "What are the three states of matter and how do they differ from one another?": "State of matter",
      "What is the role of hormones in the human body and how do they regulate various physiological processes?": "Hormone",
      "How do scientists determine the age of rocks and fossils?": "Radiometric dating",
      "What is the process of evaporation?": "Evaporation",
      "What are the different types of renewable energy and how are they used to generate electricity?": "Renewable energy",
      "What is the relationship between genetics and inheritance and how do scientists study these concepts?": "Genetics",
      "What are the different types of rocks?": "Rock (geology)",
      "What are the different types of clouds?": "Cloud",
      "What are the different types of fossils?": "Fossil",
      "What is the process of photosynthesis and how does it enable plants to produce oxygen and food?": "Photosynthesis",
      "How do scientists study the behavior of subatomic particles and what have they learned about the nature of matter and energy?": "Subatomic particle",
      "How do different types of viruses and bacteria cause disease in the body and how do scientists study and treat these illnesses?": "Infectious diseases (medical specialty)",
      "What are the different types of rocks and how do they form?": "Rock cycle"
    }
    
    # Approaches that should be evaluated (empty list means that all approaches should be evaluated)
    approaches = args.approaches
    
    file_name = f"results_factscore_{database.file_name}_{hash_id}.json"
    try:
        with open(RESULT_FOLDER + file_name, 'r') as file:
            results = json.load(file)
        processed_approaches = list(results.keys())
    except:
        results = {}
        processed_approaches = []
    
    print(approaches)
    print(processed_approaches)
    
    for _, type_group in test_steered_data.groupby("steering_type"):
        steering_type = type_group["steering_type"].values[0]
        
        if steering_type in processed_approaches:
            continue
        if approaches and steering_type not in approaches:
            continue
        print(steering_type)
        
        topics = []
        question_ids = []
        target_groups = []
        explanations = []
        scores = []

        for question, question_group in type_group.groupby("question"):
            for _, row in question_group.iterrows():
                topics.append(question2topic[row["question"]])
                question_ids.append(row["question_id"])
                target_groups.append(row["data_group"])
                explanations.append(row["answer"])
                #break
            #break
        scores = scorer.get_score(
            topics=topics,
            generations=explanations,
            gamma=0
        )["score"]

        factscores_cleaned = []

        for score in scores:
            if isinstance(score, str):
                score = 0
            factscores_cleaned.append(score)
            
        print(np.mean(factscores_cleaned))
        
        results[steering_type] = {
            "question_ids": question_ids,
            "target_groups": target_groups,
            "topics": topics,
            "explanations": explanations,
            "factscores": scores,
            "factscores_cleaned": factscores_cleaned
        }

    
    with open(RESULT_FOLDER + file_name, "w") as f:
        json.dump(results, f, indent=2)

        
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)

if __name__ == "__main__":
    main()
