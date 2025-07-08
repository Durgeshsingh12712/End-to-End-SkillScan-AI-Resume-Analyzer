from pathlib import Path
from dataclasses import dataclass

@dataclass(frozen=True)
class DataIngestionArtifact:
    data_file_path: Path