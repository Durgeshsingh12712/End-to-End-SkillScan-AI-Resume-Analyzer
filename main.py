from skillScan.pipeline import TrainingPipeline
from skillScan.logging import logger

if __name__ == "__main__":
    try:
        logger.info("Starting the training pipeline")
        training_pipeline = TrainingPipeline()
        training_pipeline.run_pipeline()
        logger.info("Training pipeline completed successfully")

    except Exception as e:
        logger.error(f"Error in training pipeline: {str(e)}")
        raise e