from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from src.report import get_text_explainer
from src.utils import construct_file_path

test_email = """
Government ID Needed for Jane Doe,
Jane Doe’s IC number 456123789012 and passport KL8901234 are required for compliance checks.
"""

MODEL_DIR = construct_file_path(r"src/model")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
classifier = pipeline("text-classification", model=str(MODEL_DIR), tokenizer=str(MODEL_DIR))

if __name__ == "__main__":
    actual_dir = construct_file_path(MODEL_DIR)
    get_text_explainer(test_email, model, tokenizer)
    [prediction] = classifier(test_email)
    print(f"Result: {'Sensitive' if prediction['label'] == 'LABEL_1' else 'Not Sensitive'}.\n")
    print("Done")