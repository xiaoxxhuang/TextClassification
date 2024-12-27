from transformers import (
    TrainingArguments,
    Trainer,
    MobileBertTokenizer,
    MobileBertForSequenceClassification,
)
from sklearn.model_selection import train_test_split


from src.filter_dataframe import filter_dataframe
from src.tokenize_create_dataset import tokenize_create_dataset
from src.report import get_classification_report

DATASET_DIR = r"src/dataset/ml_datasets.csv"
MODEL_DIR = r"src/model"
DATA_TEST_SIZE = 0.2
DATA_RANDOM_STATE = 6

TRAINING_ARGUMENT = TrainingArguments(
    output_dir="./model/training_outputs",
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    num_train_epochs=5,
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    logging_dir="./logs",
    save_total_limit=1,
    report_to=None,
)

tokenizer = MobileBertTokenizer.from_pretrained("google/mobilebert-uncased")
model = MobileBertForSequenceClassification.from_pretrained(
    "google/mobilebert-uncased", num_labels=2
)

if __name__ == "__main__":
    dataframe = filter_dataframe(DATASET_DIR)

    train_df, test_df = train_test_split(
        dataframe, test_size=DATA_TEST_SIZE, random_state=DATA_RANDOM_STATE
    )
    print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")

    train_dataset = tokenize_create_dataset(train_df, tokenizer)
    test_dataset = tokenize_create_dataset(test_df, tokenizer)

    trainer = Trainer(
        model=model,
        args=TRAINING_ARGUMENT,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
    )
    print(
        f"Training start...\n Training arguments: {TRAINING_ARGUMENT} \n This may takes 20 minutes, please wait..."
    )

    trainer.train()
    trainer.evaluate()

    print("Training completed, Saving model in progress...")

    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)

    print("Saving completed, Get Classification Report in progress...")
    predictions = trainer.predict(test_dataset)
    get_classification_report(predictions, test_df["label"])
