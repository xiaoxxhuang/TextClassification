import tarfile
from src.utils import construct_file_path

MODEL_DIR = construct_file_path(r"./src/model")
OUTPUT_TAR_GZ = construct_file_path(r"model.tar.gz")

with tarfile.open(OUTPUT_TAR_GZ, "w:gz") as tar:
    tar.add(MODEL_DIR, arcname=".")

with tarfile.open(OUTPUT_TAR_GZ, "r:gz") as tar:
    print("Contents of the tarball:")
    tar.list()

print(f"Model directory compressed to {OUTPUT_TAR_GZ}")
