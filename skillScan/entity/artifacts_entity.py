from pathlib import Path
from dataclasses import dataclass

@dataclass(frozen=True)
class DataIngestionArtifact:
    data_file_path: Path

@dataclass(frozen=True)
class DataValidationArtifact:
    validation_status: bool

@dataclass(frozen=True)
class DataTransformationArtifact:
    train_data_path: Path
    test_data_path: Path
    vectorizer_path: Path
    label_encoder_path: Path

@dataclass(frozen=True)
class ModelTrainerArtifact:
    model_path: Path
    accuracy_score: float