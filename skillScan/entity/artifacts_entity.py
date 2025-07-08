from pathlib import Path
from dataclasses import dataclass

@dataclass(frozen=True)
class DataIngestionArtifact:
    data_file_path: Path

@dataclass(frozen=True)
class DataValidationArtifact:
    validation_status: bool