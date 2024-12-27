from pathlib import Path
import sys


def construct_file_path(file_path):
    main_script_path = Path(sys.argv[0]).resolve().parent
    return (main_script_path / file_path).resolve()
