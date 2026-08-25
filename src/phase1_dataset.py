
from __future__ import annotations

import importlib

import requests
from dotenv import load_dotenv


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DATASET_URL = (
    "https://raw.githubusercontent.com/zackhuiiiii/Privacy-R1/"
    "main/med_pcd_dataset/med_pcd_all.json"
)


def load_spacy_model():
    spacy = importlib.import_module("spacy")

    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        importlib.import_module("spacy.cli").download("en_core_web_sm")
        return spacy.load("en_core_web_sm")


def load_med_pcd_dataset():
    response = requests.get(DATASET_URL, timeout=30)
    response.raise_for_status()
    dataset = response.json()

    if not isinstance(dataset, list) or not dataset:
        raise ValueError("Downloaded dataset is empty or is not a JSON list.")

    required_fields = {
        "query",
        "pii_units",
        "target_response",
    }

    for index, record in enumerate(dataset):
        missing_fields = required_fields - set(record.keys())
        if missing_fields:
            raise ValueError(
                f"Record {index} is missing fields: {sorted(missing_fields)}"
            )

    return dataset


def initialize_environment():
    print("Setting up the environment...")
    load_dotenv()

    nlp = load_spacy_model()
    print("SpaCy model loaded successfully.")

    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    feature_extractor = AutoModel.from_pretrained(MODEL_NAME)

    for param in feature_extractor.parameters():
        param.requires_grad = False

    print("MiniLM loaded with frozen weights.")

    dataset = load_med_pcd_dataset()
    print(f"Loaded {len(dataset)} Med-PCD records.")
    print("Dataset fields validated successfully for every record.")

    return nlp, tokenizer, feature_extractor, dataset


load_dataset = load_med_pcd_dataset


if __name__ == "__main__":
    initialize_environment()
