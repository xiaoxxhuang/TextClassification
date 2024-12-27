import torch


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def tokenize_create_dataset(data, tokenizer):
    encodings = tokenizer(
        list(data["email"]), padding="max_length", truncation=True, max_length=512
    )
    dataset = CustomDataset(encodings, list(data["label"]))

    return dataset
