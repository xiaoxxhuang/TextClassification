import numpy
from sklearn.metrics import classification_report
from transformers_interpret import SequenceClassificationExplainer
from .utils import construct_file_path


def get_classification_report(predictions, test_label):
    predicted_labels = numpy.argmax(predictions.predictions, axis=1)

    true_labels = test_label.values
    print(
        classification_report(
            true_labels, predicted_labels, target_names=["Non-Sensitive", "Sensitive"],labels=[0, 1]
        )
    )


def get_text_explainer(test_email, model, tokenizer):
    explanation_html_path = construct_file_path(r"src/explanation/explanation.html")
    cls_explainer = SequenceClassificationExplainer(
        model,
        tokenizer)
    cls_explainer(test_email, class_name="LABEL_1")
    cls_explainer.visualize(str(explanation_html_path))
