from shap import Explainer, plots
from src.utils import load_model_and_tokenizer
import torch
import os

test_email = """
    Government ID Needed for Jane Doe,
    Jane Doe’s IC number 456123789012 and passport KL8901234 are required for compliance checks.
    """

model, tokenizer = load_model_and_tokenizer()
def predict_proba(texts):
    inputs = tokenizer(list(texts), return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return probabilities.detach().cpu().numpy()

shap_explainer = Explainer(predict_proba, tokenizer)
shap_values = shap_explainer([test_email])
plots.text(shap_values[0, :, 1])

file = open('src/explanation/shap_explanation.html','w')
file.write(plots.text(shap_values[0, :, 0], display=False))
file.close
