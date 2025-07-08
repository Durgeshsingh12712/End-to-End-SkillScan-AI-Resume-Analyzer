import sys
from skillScan.configure import ConfigurationManager
from components import DataIngestion
from exception import SkillScanException
from skillScan.logging import logger


class TrainingPipeline:
    def __init__(self):
        pass

    def data_ingestion(self):
        try:
            # Data Ingestion
            config = ConfigurationManager()
            data_ingestion_config = config.get_data_ingestion_config()
            data_ingestion = DataIngestion(config=data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            return data_ingestion_artifact
        except Exception as e:
            raise SkillScanException(e, sys)
    
    def run_pipeline(self):
        """Run the complete training pipeline"""
        try:
            logger.info(">>>>>> Training Pipeline Started <<<<<<")
            
            # Data Ingestion
            logger.info(">>>>>> Stage 1: Data Ingestion Started <<<<<<")
            data_ingestion_artifact = self.data_ingestion()
            logger.info(">>>>>> Stage 1: Data Ingestion Completed <<<<<<")

            return {
                "data_ingestion": data_ingestion_artifact,
            }
        

        except Exception as e:
            raise SkillScanException(e, sys)