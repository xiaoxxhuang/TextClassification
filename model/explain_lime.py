from lime.lime_text import LimeTextExplainer
from src.utils import load_model_and_tokenizer
import torch
import os

test_email = """
    Government ID Needed for Jane Doe,
    Jane Doe’s IC number 456123789012 and passport KL8901234 are required for compliance checks.
    """
    
def predict_proba(texts):
    model, tokenizer = load_model_and_tokenizer()
    inputs = tokenizer(list(texts), return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return probabilities.detach().cpu().numpy()

lime_explainer = LimeTextExplainer(class_names=["Not Sensitive", "Sensitive"], bow=False)

explanation = lime_explainer.explain_instance(test_email, predict_proba, num_features=50, num_samples=100 )

output_dir = r"src/explanation"
os.makedirs(output_dir, exist_ok=True)
explanation.save_to_file(f"{output_dir}/lime_explanation.html")