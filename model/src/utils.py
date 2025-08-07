from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import sys


def construct_file_path(file_path):
    main_script_path = Path(sys.argv[0]).resolve().parent
    return (main_script_path / file_path).resolve()

def load_model_and_tokenizer():
    MODEL_DIR = construct_file_path(r"src/model")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    return model, tokenizer