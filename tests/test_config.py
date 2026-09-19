from src.config import (
    PROJECT_ROOT,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    MODEL_DIR,
)


def test_project_root_exists():
    assert PROJECT_ROOT.exists()


def test_raw_data_directory_exists():
    assert RAW_DATA_DIR.exists()


def test_processed_data_directory_exists():
    assert PROCESSED_DATA_DIR.exists()


def test_model_directory_exists():
    assert MODEL_DIR.exists()