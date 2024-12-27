import pandas
from .utils import construct_file_path


def filter_dataframe(file):
    csv_file_path = construct_file_path(file)
    print(f"The location url for the dataset: {csv_file_path}")

    dataframe = pandas.read_csv(csv_file_path, encoding="ISO-8859-1")
    filtered_data = dataframe[
        (dataframe["for_training"] == 1)
        & (dataframe["is_pii_l1"] == 0)
        & (dataframe["is_pii_l2"] == 0)
    ]
    filtered_data["email"] = filtered_data["subject"] + "\n" + filtered_data["content"]
    filtered_data["label"] = filtered_data["is_pii_l3"]

    return filtered_data[["email", "label"]]
