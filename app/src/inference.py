from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline


def load_model(model_dir):
    return pipeline("text-classification", model=model_dir, tokenizer=model_dir)


def predict(input_data, model):
    return model(input_data)