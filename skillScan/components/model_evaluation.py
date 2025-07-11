import sys
import pandas as pd

from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from skillScan.entity import ModelEvaluationConfig, ModelEvaluationArtifact
from skillScan.utils import load_object, save_json
from skillScan.exception import SkillScanException
from skillScan.logging import logger


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        try:
            logger.info("Starting Model Evaluation")

            # Load Test Data
            test_df = pd.read_csv(self.config.test_data_path)

            # Seperate Features and Target
            X_test = test_df.drop('target', axis=1)
            y_test = test_df['target']

            # Load Model
            model = load_object(self.config.model_Path)

            # Make Predictions
            y_pred = model.predict(X_test)

            # Calculate Metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')

            # Log Metrics
            logger.info(f"Model Evaluation Metrics:")
            logger.info(f"Accuracy: {accuracy}")
            logger.info(f"Precision: {precision}")
            logger.info(f"Recall: {recall}")
            logger.info(f"F1 Score: {f1}")

            # Save Metrics
            metrics = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1
            }

            save_json(Path(self.config.metric_file_name), metrics)

            return ModelEvaluationArtifact(
                accuracy_score=accuracy,
                precision_score=precision,
                recall_score=recall,
                f1_score=f1
            )
        
        except Exception as e:
            raise SkillScanException(e, sys)
        