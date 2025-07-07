import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message):')

project_name = "skillScan"

list_of_files = [
    ".github/workflows/.gitkeep",
    f"{project_name}/__init__.py",
    f"{project_name}/components/__init__.py",
    f"{project_name}/components/data_ingestion.py",
    f"{project_name}/components/data_validation.py",
    f"{project_name}/components/data_transformation.py",
    f"{project_name}/components/model_trainer.py",
    f"{project_name}/components/model_evaluation.py",
    f"{project_name}/config/__init__.py",
    f"{project_name}/config/configuration.py",
    f"{project_name}/constants/__init__.py",
    f"{project_name}/constants/constant.py",
    f"{project_name}/entity/__init__.py",
    f"{project_name}/entity/config_entity.py",
    f"{project_name}/entity/artifacts_entity.py",
    f"{project_name}/exception/__init__.py",
    f"{project_name}/exception/skillscanexception.py",
    f"{project_name}/logging/__init__.py",
    f"{project_name}/logging/skillscanlogger.py",
    f"{project_name}/utils/__init__.py",
    f"{project_name}/utils/tools.py",
    f"{project_name}/pipeline/__init__.py",
    f"{project_name}/pipeline/training_pipeline.py",
    f"{project_name}/pipeline/prediction_pipeline.py",
    "config/config.yaml",
    "config/params.yaml",
    "notebooks/research.ipynb",
    "templates/index.html",
    "templates/results.html",
    "static/css/index.css",
    "static/css/results.css",
    "static/js/index.js",
    "static/js/results.js",
    "app.py",
    "main.py",
    "requirements.txt",
    "setup.py"
]

for filepath in list_of_files:
    filepath = Path(filepath)

    filedir, filename = os.path.split(filepath)
    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating Directory: {filedir} for the file {filename}")

    if(not os.path.exists(filename)) or (os.path.getsize(filename) == 0):
        with open(filepath, "w") as f:
            pass
        logging.info(f"Creating Empty File: {filename}")
    
    else:
        logging.info(f"{filename} is alreaady Created")