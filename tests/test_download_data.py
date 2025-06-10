from pathlib import Path
from unittest import mock

import pytest
from hydra import compose, initialize

from argueflow.utils.dvc_utils import download_data


@pytest.fixture
def cfg():
    with initialize(config_path="../configs", version_base="1.3"):
        cfg = compose(config_name="config")
    return cfg


def test_download_skipped_if_files_exist(tmp_path, cfg, monkeypatch):
    """
    Test that `download_data` does not trigger a DVC pull
    if all required data files already exist locally.

    Mocks file creation and patches `subprocess.run` to assert
    it is not called
    """
    cfg.data.raw_train_csv = str(tmp_path / "data/raw/train.csv")
    cfg.data.processed_data_path = str(tmp_path / "data/processed/train_prepared.csv")

    (Path(cfg.data.raw_train_csv)).parent.mkdir(parents=True, exist_ok=True)
    (Path(cfg.data.raw_train_csv)).write_text("id,text\n1,Hello")

    (Path(cfg.data.processed_data_path)).parent.mkdir(parents=True, exist_ok=True)
    (Path(cfg.data.processed_data_path)).write_text("id,label\n1,Effective")

    with mock.patch("subprocess.run") as mock_subproc:
        download_data(cfg)
        mock_subproc.assert_not_called()


def test_download_called_if_files_missing(cfg, monkeypatch):
    """
    Test that `download_data` triggers a DVC pull
    when required data files are missing.

    Mocks `Path.exists` to always return False and asserts
    that `subprocess.run` is called with the correct DVC command.
    """
    cfg.data.raw_train_csv = "nonexistent/raw/train.csv"
    cfg.data.processed_data_path = "nonexistent/processed/train_prepared.csv"

    monkeypatch.setattr(Path, "exists", lambda self: False)

    with mock.patch("subprocess.run") as mock_subproc:
        download_data(cfg)
        mock_subproc.assert_called_once_with(["dvc", "pull"], check=True)
