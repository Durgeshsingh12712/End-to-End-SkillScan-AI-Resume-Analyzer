import sys

from skillScan.configure import ConfigurationManager
from skillScan.exception import SkillScanException
from skillScan.logging import logger

from skillScan.components import (
    DataIngestion,
    DataValidation
)


class TrainingPipeline:
    def __init__(self):
        pass

    def data_ingestion(self):
        try:
            config = ConfigurationManager()
            data_ingestion_config = config.get_data_ingestion_config()
            data_ingestion = DataIngestion(config=data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            return data_ingestion_artifact
        except Exception as e:
            raise SkillScanException(e, sys)
    
    def data_validation(self):
        try:
            Config = ConfigurationManager()
            data_validation_config = Config.get_data_validation_config()
            data_validation = DataValidation(config=data_validation_config)
            data_validation_artifact = data_validation.initiate_data_validation()
            return data_validation_artifact
        except Exception as e:
            raise SkillScanException(e, sys)
    
    def run_pipeline(self):
        """Run the complete training pipeline"""
        try:
            logger.info(">>>>>>> Training Pipeline Started <<<<<<<")
            
            # Data Ingestion
            logger.info(">>>>>>> Stage 1: Data Ingestion Started <<<<<<<")
            data_ingestion_artifact = self.data_ingestion()
            logger.info(">>>>>>> Stage 1: Data Ingestion Completed <<<<<<<")

            # Data Validation
            logger.info(">>>>>>> Stage 2: Data Validation Started <<<<<<<")
            data_validation_artifact = self.data_validation()
            logger.info(">>>>>>> Stage 2: Data Validation Completed <<<<<<<")

            return {
                "data_ingestion": data_ingestion_artifact,
                "data_validation": data_validation_artifact,
            }
        

        except Exception as e:
            raise SkillScanException(e, sys)