import os, sys, yaml, pickle, json
from pathlib import Path
from typing import Any
from ensure import ensure_annotations

from skillScan.exception import SkillScanException
from skillScan.logging import logger


@ensure_annotations
def read_yaml(path_to_yaml: Path) -> dict:
    """Read Yaml file and return its content as dict"""
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"YAML file: {path_to_yaml} loaded successfully")
            return content
    
    except Exception as e:
        raise SkillScanException(e, sys)


@ensure_annotations
def create_directories(path_to_directories: list, verbose= True):
    """Create list of directories"""
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"Created Directory at: {path}")


@ensure_annotations
def save_json(path: Path, data: dict):
    """Save json Data"""
    try:
        with open(path, "w") as f:
            json.dump(data, f, indent=4)

        logger.info(f"Json file saved at: {path}")
    
    except Exception as e:
        raise SkillScanException(e, sys)

@ensure_annotations
def load_json(path: Path) -> dict:
    """load json files data"""
    with open(path) as f:
        content = json.load(f)
    
    logger.info(f"Json file loaded Successfully from: {path}")
    return content

@ensure_annotations
def get_size(path: Path) -> str:
    """get size in KB"""
    size_in_kb = round(os.path.getsize(path)/1024)
    return f"~ {size_in_kb} KB"

def save_object(file_path: Path, obj: Any):
    """Save object as pickle file"""
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)   
        logger.info(f"Object saved at {file_path}")
    
    except Exception as e:
        raise SkillScanException(e, sys)
    

def load_object(file_path: Path) -> Any:
    """Load object from pickle file"""
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise SkillScanException(e, sys)