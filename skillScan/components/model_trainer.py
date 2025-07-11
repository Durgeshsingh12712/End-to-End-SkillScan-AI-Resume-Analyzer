import sys
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report

from skillScan.entity import ModelTrainerConfig, ModelTrainerArtifact
from skillScan.utils import save_object
from skillScan.exception import SkillScanException
from skillScan.logging import logger

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def get_best_models(self):
        """Define the best models with optimized hyperparameters"""
        model = {
            'SVM_Optimized': OneVsRestClassifier(
                SVC(
                    C=10, 
                    kernel='rbf', 
                    gamma='scale',
                    probability=True,
                    random_state=42
                )
            ),

            'RandomForest_Optimized': OneVsRestClassifier(
                RandomForestClassifier(
                    n_estimators=100,
                    max_depth=15,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    max_features='sqrt',
                    random_state=42,
                    n_jobs=-1
                )
            ),

            'XGBoost': OneVsRestClassifier(
                XGBClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth = 6,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    n_jobs=-1,
                    eval_metric='mlogloss'
                )
            ),
            
            'LightGBM': OneVsRestClassifier(
                LGBMClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=6,
                    num_leaves=31,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    n_jobs=-1,
                    verbose = -1
                )
            ),

            'GradientBoosting': OneVsRestClassifier(
                GradientBoostingClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=6,
                    random_state=42
                )
            )
        }

        return model
    

    def create_ensemble_model(self, models_dict, X_train, y_train, X_test, y_test):
        """Create an ensemble of best performing models"""

        model_scores = {}
        trained_models = {}

        #Train and evaluate each model
        for name, model in models_dict.items():
            try:
                logger.info(f"Training {name}")
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                model_scores[name] = accuracy
                trained_models[name] = model
                logger.info(f"{name} Accuracy: {accuracy:.4f}")
            except Exception as e:
                logger.warning(f"Failed to train {name}: {str(e)}")
                continue
        
        if not model_scores:
            raise Exception("All Models Failed to Train")
        
        # Select Top 3 Model for ensemble
        top_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        logger.info(f"Top 3 Models: {[model[0] for model in top_models]}")

        # Create ensemble of Top Models
        ensemble_models = [(name, trained_models[name]) for name, _ in top_models]

        # Voting Classifier(Soft voting for probability-besed confidence)
        ensemble_model = VotingClassifier(
            estimators=ensemble_models,
            voting='soft'
        )

        # Train Ensemble
        logger.info("Training Ensemble Model")
        ensemble_model.fit(X_train, y_train)

        # Evaluate ensemble
        y_pred_ensemble = ensemble_model.predict(X_test)
        ensemble_accuracy = accuracy_score(y_test, y_pred_ensemble)
        logger.info(F"Ensemble Accuracy: {ensemble_accuracy:.4f}")

        # Return best Individual Model and Ensemble
        best_individual = max(model_scores.items(), key=lambda x: x[1])

        return {
            'best_individual': {
                'name': best_individual[0],
                'model': trained_models[best_individual[0]],
                'accuracy': best_individual[1]
            },

            'ensemble': {
                'name': 'Ensemble_Top3',
                'model': ensemble_model,
                'accuracy': ensemble_accuracy
            },

            'all_scores': model_scores
        }


    def quick_hyperparameter_tuning(self, X_train, y_train):
        "Ultra-Fast Hyperparameter tuning with minimal parameter search"""

        xgb_param_grid = {
            'estimator__n_estimators': [50, 100],
            'estimator__learning_rate': [0.1, 0.2],
            'estimator__max_depth': [4, 6]
        }
        
        rf_param_grid = {
            'estimator__n_estimators': [50, 100],
            'estimator__max_depth': [10, 15],
            'estimator__min_samples_split': [2, 5]
        }
        
        results = {}

        # XGBoost Tuning
        try:
            xgb_model = OneVsRestClassifier(XGBClassifier(random_state=42, eval_metric='mlogloss'))
            xgb_search = RandomizedSearchCV(
                xgb_model,
                xgb_param_grid,
                n_iter=4,
                cv=2,
                scoring='accuracy',
                n_jobs=-1,
                random_state=42
            )
            logger.info("Performing Quick XGBoost hyperparameter tuning...")
            xgb_search.fit(X_train, y_train)

            results.update({
                'xgb_best': xgb_search.best_estimator_,
                'xgb_score': xgb_search.best_score_,
                'xgb_params': xgb_search.best_params_
            })
        except Exception as e:
            logger.warning(f"XGBoost Tunning Failed: {str(e)}")

        
        # Random Forest Tuning
        try:
            rf_model = OneVsRestClassifier(RandomForestClassifier(random_state=42))
            rf_search = RandomizedSearchCV(
                rf_model,
                rf_param_grid,
                n_iter=4,
                cv = 2,
                scoring = 'accuracy',
                n_jobs=-1,
                random_state=42
            )
            logger.info("Performaing Quick Random Forest Hyperparameter tuning...")
            rf_search.fit(X_train, y_train)

            results.update({
                'rf_best': rf_search.best_estimator_,
                'rf_score': rf_search.best_score_,
                'rf_params': rf_search.best_params_
            })
        except Exception as e:
            logger.warning(f"Random Forest tuning Failed: {str(e)}")

        return results
    
    def fast_hyperparameter_tuning(self, X_train, y_train):
        """Fast Hyperparameter tuning using RandomizedSearchCV"""

        xgb_param_grid = {
            'estimator__n_estimators': [50, 100, 150],
            'estimator__learning_rate': [0.1, 0.15, 0.2],
            'estimator__max_depth': [4, 6, 8],
            'estimator__subsample': [0.8, 0.9]
        }
        
        rf_param_grid = {
            'estimator__n_estimators': [50, 100, 150],
            'estimator__max_depth': [10, 15, 20],
            'estimator__min_samples_split': [2, 5],
            'estimator__max_features': ['sqrt', 'log2']
        }
        
        
        lgb_param_grid = {
            'estimator__n_estimators': [50, 100, 150],
            'estimator__learning_rate': [0.1, 0.15, 0.2],
            'estimator__max_depth': [4, 6, 8],
            'estimator__num_leaves': [15, 31, 63]
        }
        
        results = {}

        # XGBoost
        try:
            xgb_model = OneVsRestClassifier(XGBClassifier(random_state=42, eval_metric='mlogloss'))
            xgb_search = RandomizedSearchCV(
                xgb_model,
                xgb_param_grid,
                n_iter=10,
                cv=3,
                scoring='accuracy',
                n_jobs=-1,
                verbose=1,
                random_state=42
            )
            logger.info("Performing Fast XGBoost hyperparameter tuning (RandomizedSearch)...")
            xgb_search.fit(X_train, y_train)

            results.update({
                'xgb_best': xgb_search.best_estimator_,
                'xgb_score': xgb_search.best_score_,
                'xgb_params': xgb_search.best_params_
            })
        except Exception as e:
            logger.warning(f"XGBoost tuning Failed: {str(e)}")

        
        #Random Forest
        try:
            rf_model = OneVsRestClassifier(RandomForestClassifier(random_state=42))
            rf_search = RandomizedSearchCV(
                rf_model,
                rf_param_grid,
                n_iter=10,
                cv=3,
                scoring='accuracy',
                n_jobs=-1,
                verbose=1,
                random_state=42
            )
            logger.info("Performing Fast Random Forest Hyperparameter tuning (RandomizedSearch)...")
            rf_search.fit(X_train, y_train)

            results.update({
                'rf_best': rf_search.best_estimator_,
                'rf_score': rf_search.best_score_,
                'rf_params': rf_search.best_params_
            })
        except Exception as e:
            logger.warning(f"RandomForest Tuning Failed: {str(e)}")

        # LightGBM
        try:
            lgb_model = OneVsRestClassifier(LGBMClassifier(random_state=42, verbose=-1))
            lgb_search = RandomizedSearchCV(
                lgb_model,
                lgb_param_grid,
                n_iter=10,
                cv=3,
                scoring='accuracy',
                n_jobs=-1,
                verbose=1,
                random_state=42
            )
            
            logger.info("Performing FAST LightGBM hyperparameter tuning (RandomizedSearch)...")
            lgb_search.fit(X_train, y_train)
            
            results.update({
                'lgb_best': lgb_search.best_estimator_,
                'lgb_score': lgb_search.best_score_,
                'lgb_params': lgb_search.best_params_
            })
        except Exception as e:
            logger.warning(f"LightGBM tuning failed: {str(e)}")

        return results

    def comprehensive_hyperparameter_tuning(self, X_train, y_train):
        """Original Comprehensive Hyperparameter Tuning Methed"""

        xgb_param_grid = {
            'estimator__n_estimators': [100, 200],
            'estimator__learning_rate': [0.05, 0.1, 0.15],
            'estimator__max_depth': [6, 8, 10],
            'estimator__subsample': [0.8, 0.9],
            'estimator__colsample_bytree': [0.8, 0.9]
        }
        
        rf_param_grid = {
            'estimator__n_estimators': [100, 200],
            'estimator__max_depth': [15, 20, None],
            'estimator__min_samples_split': [2, 5, 10],
            'estimator__min_samples_leaf': [1, 2, 4],
            'estimator__max_features': ['sqrt', 'log2']
        }
        
        lgb_param_grid = {
            'estimator__n_estimators': [100, 200],
            'estimator__learning_rate': [0.05, 0.1, 0.15],
            'estimator__max_depth': [6, 8, 10],
            'estimator__num_leaves': [31, 63, 127],
            'estimator__subsample': [0.8, 0.9]
        }
        
        results = {}

        #XGBoost
        try:
            xgb_model = OneVsRestClassifier(XGBClassifier(random_state=42, eval_metric='mlogloss'))
            xgb_grid = GridSearchCV(
                xgb_model,
                xgb_param_grid,
                cv=5,
                scoring='accuracy',
                n_jobs=-1,
                verbose=1
            )
            logger.info("Performing Comprehensive XGBoost Hyperparameter Tuning (GridSearch)...")
            xgb_grid.fit(X_train,y_train)

            results.update({
                'xgb_best': xgb_grid.best_estimator_,
                'xgb_score': xgb_grid.best_score_,
                'xgb_params': xgb_grid.best_params_
            })
        except Exception as e:
            logger.warning(f"XGBoost Tuning Failed: {str(e)}")
        
        # Random Forest
        try:
            rf_model = OneVsRestClassifier(RandomForestClassifier(random_state=42))
            rf_grid = GridSearchCV(
                rf_model,
                rf_param_grid,
                cv=5,
                scoring='accuracy',
                n_jobs=-1,
                verbose=1
            )
            logger.info("Performing Comprehensive Random Forest Hyperparameter tuning (GridSearch)...")
            rf_grid.fit(X_train, y_train)

            results.update({
                'rf_best': rf_grid.best_estimator_,
                'rf_score': rf_grid.best_score_,
                'rf_params': rf_grid.best_params_
            })
        except Exception as e:
            logger.warning(f"XGBoost Tuning Failed: {str(e)}")
        
        # LightGBM
        try:
            lgb_model = OneVsRestClassifier(LGBMClassifier(random_state=42, verbose=-1))
            lgb_grid = GridSearchCV(
                lgb_model,
                lgb_param_grid,
                cv=5,
                scoring='accuracy',
                n_jobs=-1,
                verbose=1
            )
            logger.info("Performing Comprehensive LightGBM hyperparameter Tuning (GridSearch)...")
            lgb_grid.fit(X_train, y_train)

            results.update({
                'lgb_best': lgb_grid.best_estimator_,
                'lgb_score': lgb_grid.best_score_,
                'lgb_params': lgb_grid.best_params_
            })
        except Exception as e:
            logger.warning(f"LightGBM Tuning failed: {str(e)}")
        
        return results
    
    def evaluate_tuned_models(self, tuned_results, X_test, y_test):
        """Evaluate all tuned models and return their scores"""
        tuned_models = {}
        tuned_scores = {}

        model_mapping = {
            'xgb_best': 'XGB_Tuned',
            'rf_best': 'RF_Tuned',
            'lbg_best': 'LGB_Tuned'
        }

        for key, model_name in model_mapping.items():
            if key in tuned_results:
                try:
                    model = tuned_results[key]
                    y_pred = model.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)
                    tuned_models[model_name] = model
                    tuned_scores[model_name] = accuracy
                    logger.info(f"{model_name} Accuracy: {accuracy:.4f}")
                except Exception as e:
                    logger.warning(f"Failed to Evaluate {model_name}: {str(e)}")
                
        return tuned_models, tuned_scores
    
    def save_models_results(self, final_model, best_model_name, best_accuracy,
                            all_models, tuning_mode, tuned_results,
                            y_test, y_pred_final):
        """Save the Final Model and All Related Metadata"""

        model_path = Path(self.config.root_dir) / self.config.model_name
        save_object(model_path, final_model)
        logger.info(f"Model Saved to: {model_path}")

        report = classification_report(y_test, y_pred_final)

        #Save Model MetaData
        metadata = {
            'best_model_name': best_model_name,
            'accuracy': best_accuracy,
            'all_model_scores': all_models,
            'classification_report': report,
            'tuning_mode': tuning_mode,
            'tuned_params': tuned_results if tuned_results else None,
            'model_path': str(model_path)
        }

        metadata_path = Path(self.config.root_dir) / "model_metadata.pkl"
        save_object(metadata_path, metadata)
        logger.info(f"MetaData Saved to: {metadata}")

        return model_path, metadata
    

    def initiate_model_trainer(self, tuning_mode = 'fast') -> ModelTrainerArtifact:
        """
        Main Methed to Train Models with different tuning strategies

        Tuning_Mode Option:
            'none' : Skip Hyperparameter Tuning Entirely (Fastest)
            'quick' : Ultra-Fast tuning with mininmal parameters (~2-5 Minutes)
            'fast' : Fast Tuning with RandomizedSearchCV (~10-20 Minutes)
            'comprehensive' : Full Comprehensive Tuning (Slowest, 1-2 Hours)
        """

        try:
            logger.info(f"Starting Advanced Model Training with Tuning Mode: {tuning_mode}")

            # Load train and Test Data
            train_df = pd.read_csv(self.config.train_data_path)
            test_df = pd.read_csv(self.config.test_data_path)

            # Seperate Features and Target
            X_train = train_df.drop('target', axis=1)
            y_train = train_df['target']
            X_test = test_df.drop('target', axis=1)
            y_test = test_df['target']

            logger.info(f"Train Data Shape: {X_train.shape}")
            logger.info(f"Test Data Shape: {X_test.shape}")
            logger.info(f"Number of Classes: {len(np.unique(y_train))}")

            # Optimized MOdel
            models = self.get_best_models()

            # Train All Model and Create Ensemble
            results = self.create_ensemble_model(models, X_train, y_train, X_test, y_test)

            # Initiate Variables for tuned Models
            tuned_results = None
            tuned_models = {}
            tuned_scores = {}

            # Choose Tuning Strategy Based on Mode
            if tuning_mode == 'none':
                logger.info("Skipping Hyperparameter Tuning for Fastest Training")

            elif tuning_mode == 'quick':
                logger.info("Starting Quick Hyperparameter Tuning (2-5 Minutes)")
                tuned_results = self.quick_hyperparameter_tuning(X_train, y_train)
                tuned_models, tuned_scores = self.evaluate_tuned_models(tuned_results, X_test, y_test)

            elif tuning_mode == 'fast':
                logger.info("Starting Fast Hyperparameter Tuning (10-20 Minutes)")
                tuned_results = self.fast_hyperparameter_tuning(X_train, y_train)
                tuned_models, tuned_scores = self.evaluate_tuned_models(tuned_results, X_test, y_test)
            
            elif tuning_mode == 'comprehensive':
                logger.info("Starting Comprehensive Hyperparameter Tuning (1-2 Hours)")
                tuned_results = self.comprehensive_hyperparameter_tuning(X_train, y_train)
                tuned_models, tuned_scores = self.evaluate_tuned_models(tuned_results, X_test, y_test)

            else:
                raise ValueError(f"Invalid tuning_mode: {tuning_mode}. Choose from: 'none', 'quick', 'fast., 'comprehensive'")
            
            # Combine All Model Score
            all_models = {
                **{k: v for k, v in results['all_scores'].items()},
                **tuned_scores,
                'Ensemble': results['ensemble']['accuracy']
            }

            # Find The absolute best model
            best_model_name = max(all_models.items(), key=lambda x: x[1])[0]
            best_accuracy = all_models[best_model_name]

            # Select The Best Model
            if best_model_name == 'Ensemble':
                final_model = results['ensemble']['model']
            elif best_model_name in tuned_models:
                final_model = tuned_models[best_model_name]
            else:
                final_model = models[best_model_name]
                # Ensure it's Trained
                if not hasattr(final_model, 'classes_'):
                    final_model.fit(X_train, y_train)
                
            logger.info(f"Best Model Selected: {best_model_name}")
            logger.info(f"Best Accuracy: {best_accuracy:.4f}")

            #Generate Prediction with the Final Model
            y_pred_final = final_model.predict(X_test)

            # Print Detailed Results
            logger.info("\n" + "="*50)
            logger.info("Final Results Summary")
            logger.info("="*50)
            logger.info(f"Best Model: {best_model_name}")
            logger.info(f"Test Accuracy: {best_accuracy:.4f}")
            logger.info(f"Tuning Mode: {tuning_mode}")
            
            # Print All Model Scores
            logger.info("\nALL Model Scores:")
            for name, score in sorted(all_models.items(), key=lambda x: x[1], reverse=True):
                logger.info(f" {name}: {score:.4f}")
            
            # Print Classification Report
            logger.info("\nDetailed Classifiacation Report:")
            logger.info(classification_report(y_test, y_pred_final))

            # Save Model and Metadata
            model_path, metadata = self.save_models_results(
                final_model,
                best_model_name,
                best_accuracy,
                all_models,
                tuning_mode,
                tuned_results,
                y_test,
                y_pred_final
            )

            logger.info(f"\nModel Training Completed Successfully")
            logger.info(f"Final Model Saved to: {model_path}")

            return ModelTrainerArtifact(
                model_path=model_path,
                accuracy_score=best_accuracy
            )

        except Exception as e:
            logger.error(f"Error in Model Training: {str(e)}")
            raise SkillScanException(e, sys)

# Legacy method for backward compatibility
    def hyperparameter_tuning(self, X_train, y_train):
        """Legacy method - calls comprehensive tuning for backward compatibility"""
        logger.warning("Using legacy hyperparameter_tuning method. Consider using initiate_model_trainer with tuning_mode parameter.")
        return self.comprehensive_hyperparameter_tuning(X_train, y_train)
