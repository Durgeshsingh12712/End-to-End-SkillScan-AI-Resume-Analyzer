import sys

from skillScan.configure import ConfigurationManager
from skillScan.exception import SkillScanException
from skillScan.logging import logger

from skillScan.components import (
    DataIngestion,
    DataValidation,
    DataTransformation,
    ModelTrainer,
    ModelEvaluation
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
            config = ConfigurationManager()
            data_validation_config = config.get_data_validation_config()
            data_validation = DataValidation(config=data_validation_config)
            data_validation_artifact = data_validation.initiate_data_validation()
            return data_validation_artifact
        except Exception as e:
            raise SkillScanException(e, sys)
    
    def data_transformation(self):
        try:
            config = ConfigurationManager()
            data_transformation_config = config.get_data_transformation_config()
            data_transformation = DataTransformation(config = data_transformation_config)
            data_transformation_artifact = data_transformation.initiate_data_transformation()
            return data_transformation_artifact
        except Exception as e:
            raise SkillScanException(e, sys)
    
    def model_trainer(self):
        try:
            config = ConfigurationManager()
            model_trainer_config = config.get_model_trainer_config()
            model_trainer = ModelTrainer(config=model_trainer_config)
            model_trainer_artifact = model_trainer.initiate_model_trainer()
            return model_trainer_artifact
        except Exception as e:
            raise SkillScanException(e, sys)
    
    def model_evaluation(self):
        try:
            config = ConfigurationManager()
            model_evaluation_config = config.get_model_evaluation_config()
            model_evaluation = ModelEvaluation(config=model_evaluation_config)
            model_evaluation_artifact = model_evaluation.initiate_model_evaluation()
            return model_evaluation_artifact
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

            # Data Transformation
            logger.info(">>>>>>> Stage 3: Data Transformation Started <<<<<<<")
            data_transformation_artifact = self.data_transformation()
            logger.info(">>>>>>> Stage 3: Data Transformation Completed <<<<<<<")

            # Model Trainer
            logger.info(">>>>>>> Stage 4: Model Trainer Started <<<<<<<")
            model_trainer_artifact = self.model_trainer()
            logger.info(">>>>>>> Stage 4: Model Trainer Completed <<<<<<<")

            # Model Evaluation
            logger.info(">>>>>>> Stage 5: Model Evaluation Started <<<<<<<")
            model_evaluation_artifact = self.model_evaluation()
            logger.info(">>>>>>> Stage 5: Model Evaluation Completed <<<<<<<")

            return {
                "data_ingestion": data_ingestion_artifact,
                "data_validation": data_validation_artifact,
                "data_transformation": data_transformation_artifact,
                "model_trainer": model_trainer_artifact,
                "model_evaluation": model_evaluation_artifact
            }
        

        except Exception as e:
            raise SkillScanException(e, sys)
