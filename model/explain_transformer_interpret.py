from transformers_interpret import SequenceClassificationExplainer
from src.utils import load_model_and_tokenizer

test_email = """
    Government ID Needed for Jane Doe,
    Jane Doe’s IC number 456123789012 and passport KL8901234 are required for compliance checks.
    """

model, tokenizer = load_model_and_tokenizer()
cls_explainer = SequenceClassificationExplainer(
    model,
    tokenizer)
word_attributions = cls_explainer(test_email, class_name="LABEL_1")
cls_explainer.visualize("src/explanation/ti_explanation.html")