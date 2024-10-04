import argparse
from enum import Enum
import os
from pathlib import Path
import sys
import zipfile

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(project_root)

from dotenv import load_dotenv
import wandb
from utils import Constants

load_dotenv()

class Mode(Enum):
    DOWNLOAD = "download"
    UPLOAD = "upload"

class DatasetName(Enum):
    NEXET = "nexet"

def manage_db(dataset_type: DatasetName, mode: Mode, local_db_root: str = ""):

	if dataset_type == DatasetName.NEXET:
		local_db_filepath = Path(local_db_root) / Constants.DATASET_FILEPATH 
	wb_db_name = local_db_filepath.stem

	if mode == Mode.UPLOAD:
		_push_db_to_wandb(local_db_filepath, wb_db_name)
	elif mode == Mode.DOWNLOAD:
		_pull_db_from_wandb(wb_db_name, local_db_filepath.parent)


def _push_db_to_wandb(local_db_filepath: Path, wb_db_name: str):
    """Push a local file to WandB as an artifact.

    This operation is logged in WandB dashboard as a run.
    If the artifact already exists in WandB, a new version will be
    automatically created.

    Args:
        local_db_filepath (Path): Path to the local file.
        wb_db_name (str): Name of the artifact created in WandB.
    """

    assert local_db_filepath.exists(), f"The provided local DB path does not exist: '{str(local_db_filepath)}'"

    if local_db_filepath.is_file():
        file_to_upload = local_db_filepath
    elif local_db_filepath.is_dir():
        zip_filename = local_db_filepath.with_suffix('.zip')
        with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(local_db_filepath):
                for file in files:
                    file_path = Path(root) / file
                    zipf.write(file_path, file_path.relative_to(local_db_filepath))
        file_to_upload = zip_filename


    wb_run = wandb.init(
        project=Constants.WB_PROJECT,
        name=wb_db_name,
        job_type=Constants.WB_DB_UPLOAD_JOB,
    )

    with wb_run:
        wb_db = wandb.Artifact(wb_db_name, type=Constants.WB_DB_ARTIFACT_TYPE)
        wb_db.add_file(local_path=str(file_to_upload), skip_cache=True)
        wb_run.log_artifact(wb_db)

    if file_to_upload.suffix == ".zip":
        file_to_upload.unlink()


def _pull_db_from_wandb(wb_db_name: str, local_db_path: Path):
    """Pulls an artifact from WandB.

    This process is not logged in the WandB dashboard. If
    multiple versions of the same artifact exist, only the latest
    will be pulled.

    Args:
        wb_db_name (str): Name of the artifact in WandB.
        local_db_path (Path): Path to the local file dir.
    """
    assert not local_db_path.suffix, f"'{str(local_db_path)}' should be a dir."
    local_db_path.mkdir(parents=True, exist_ok=True)

    api = wandb.Api()
    artifact_full_path = f"{Constants.WB_PROJECT}/{wb_db_name}:latest"
    artifact = api.artifact(artifact_full_path)
    artifact.download(root=local_db_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Upload and download datasets to and from WandB.",
    )
    parser.add_argument(
        "-m",
        "--mode",
        choices=[m.value for m in Mode],
        required=True,
    )
    parser.add_argument(
        "-db",
        "--dataset_type",
        choices=[t.value for t in DatasetName],
        required=True,
    )
    args = parser.parse_args()
    manage_db(DatasetName(args.dataset_type), Mode(args.mode))

# E.g., $python src/dataset/manage_wb_dataset.py -m upload -db nexet
# E.g., $python src/dataset/manage_wb_dataset.py -m download -db nexet