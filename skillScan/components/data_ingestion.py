import os, sys, zipfile
from urllib import request

from entity import DataIngestionConfig
from entity import DataIngestionArtifact
from exception import SkillScanException
from skillScan.logging import logger


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self):
        if not os.path.exists(self.config.local_data_file):
            filename, headers = request.urlretrieve(
                url=self.config.source_URL,
                filename=self.config.local_data_file
            )
            logger.info(f"{filename} downloaded! with following info: \n{headers}")
        else:
            logger.info(f"File already exists")
    
    def extract_zip_file(self):
        """Extract zip file"""
        try:
            unzip_path = self.config.unzip_dir
            os.makedirs(unzip_path, exist_ok=True)
            with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
                zip_ref.extractall(unzip_path)
        except Exception as e:
            logger.info("File is not a zip file, proceeding with CSV")

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logger.info("Starting data ingestion")
            self.download_file()
            self.extract_zip_file()
            
            data_ingestion_artifact = DataIngestionArtifact(
                data_file_path=self.config.local_data_file
            )
            
            logger.info("Data ingestion completed successfully")
            return data_ingestion_artifact
            
        except Exception as e:
            raise SkillScanException(e, sys)