import sys, re
import numpy as np
import pandas as pd

from pathlib import Path

from skillScan.exception import SkillScanException
from skillScan.utils import load_object
from skillScan.logging import logger


class PredictionPipeline:
    def __init__(self):
        """Initialize the Prediction Pipeline with proper error handling"""
        try:
            logger.info("Initializing Prediction Pipeline...")

            self.vectorizer_path = Path("artifacts/data_transformation/vectorizer.pkl")
            self.label_encoder_path = Path("artifacts/data_transformation/label_encoder.pkl")
            self.model_path = Path("artifacts/model_trainer/model.pkl")
            self.metadata_path = Path("artifacts/model_trainer/model_metadata.pkl")

            # Check If Files exist
            self.check_files_exist()

            # Load Model and Components
            self.load_components()

            logger.info("Prediction Pipeline Initialized Successfully")
        
        except Exception as e:
            logger.error(f"Error initializing Prediction Pipeline: {str(e)}")
            raise SkillScanException(e, sys)
    
    def check_files_exist(self):
        """Check if all required files exist"""
        required_files = {
            'vectorizer': self.vectorizer_path,
            'label_encoder': self.label_encoder_path,
            'model': self.model_path,
            'metadata': self.metadata_path
        }

        missing_files = []
        for name, path in required_files.items():
            if not path.exists():
                missing_files.append(f"{name}: {path}")
        
        if missing_files:
            error_msg = f"Missing required files: {', '.join(missing_files)}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
    def load_components(self):
        """Load all model components with validation"""
        try:
            logger.info(f"Loading Vectorizer from: {self.vectorizer_path}")
            self.vectorizer = load_object(self.vectorizer_path)

            # Validate Vectorizer
            if not hasattr(self.vectorizer, 'idf_'):
                raise ValueError("Vectorizer is not fittted. The 'idf_' attribute is missing.")
            
            if not hasattr(self.vectorizer, 'vocabulary_'):
                raise ValueError("Vectorizer is not fitted. The 'vocabulary_' attribute is missing.")
            
            logger.info(f"Vectorizer Loaded Successfully with Vacabulary size: {len(self.vectorizer.vocabulary_)}")

            logger.info(f"Loading Label Encoder from: {self.label_encoder_path}")
            self.label_encoder = load_object(self.label_encoder_path)

            logger.info(f"Loading Model from: {self.model_path}")
            self.model = load_object(self.model_path)

            logger.info(f"Loading MetaData from: {self.metadata_path}")
            self.metadata = load_object(self.metadata_path)

        except Exception as e:
            logger.error(f"Error Loading Components: {str(e)}")
            raise SkillScanException(e, sys)
    
    def clean_resume(self, resume_text):
        """Enhanced resume text cleaning - Match Exactly with Training"""
        try:
            if not resume_text or not isinstance(resume_text, str):
                return ""
            
            # This should match exactly with your training  clean_resume methed
            resume_text = re.sub('http\S+\s*', ' ', resume_text)  # remove URLs
            resume_text = re.sub('RT|cc', ' ', resume_text)  # remove RT and cc
            resume_text = re.sub('#\S+', '', resume_text)  # remove hashtags
            resume_text = re.sub('@\S+', '  ', resume_text)  # remove mentions
            resume_text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', resume_text)
            resume_text = re.sub(r'[^\x00-\x7f]', r' ', resume_text)
            resume_text = re.sub('\s+', ' ', resume_text)  # remove extra whitespace

            return resume_text.strip()
        
        except Exception as e:
            logger.error(f"Error Cleaning resume text: {str(e)}")
            return ""
        
    def get_confidence_interpretation(self, confidence, gap=None):
        """Interpret Confidence Levels with Context"""
        if confidence > 0.8:
            return "Very High Confidence - Strong Match Found"
        elif confidence > 0.6:
            return "High Confidence - Good Match"
        elif confidence > 0.4:
            return "Medium Confidence - Reasonable Match"
        elif confidence > 0.25:
            return "Low Confidence - Weak Match , consider Manual Review"
        elif confidence > 0.15:
            return  "Very Low Confidence - Highly Uncertain, manual review recommeded"
        else:
            return "Extremely Low Confidence - No Clear Match, requires human judgement"
        
    def predict_with_enhanced_confidence(self, resume_text, debug=False, return_dict=False):
        """Enhanced Prediction with detailed Confidence Analysis"""
        try:
            logger.info("Starting Enhanced Prediction with Confidence Calculation")

            # Validate Input
            if not resume_text or not isinstance(resume_text, str):
                result_dict = {
                    'status': 'error',
                    'error': 'Invalid input: resume text must be a non-empty string',
                    'prediction': None,
                    'confidence': 0.0,
                    'confidence_level': 'Error',
                    'confidence_interpretation': 'Error',
                    'is_reliable': False,
                    'all_probabilities': {}
                }
                return result_dict if return_dict else PredictionResult(result_dict)
            
            # Clean the Resume Text
            cleaned_text = self.clean_resume(resume_text)

            if len(cleaned_text.strip()) < 10:
                result_dict = {
                    'status': 'errror',
                    'error': 'Resume Text too short for reliable prediction',
                    'prediction': None,
                    'confidence': 0.0,
                    'confidence_level': 'Error',
                    'confidence_interpretation': 'Error',
                    'is_reliable': False,
                    'all_probabilities': {}
                }
                return result_dict if return_dict else PredictionResult(result_dict)
            
            if debug:
                logger.info(f"Original Text Length: {len(resume_text)}")
                logger.info(f"Cleaned Text Length: {len(cleaned_text)}")
                logger.info(f"Cleaned Text Sample: {cleaned_text[:200]}...")

            # Vectorize The Text
            try:
                vectorized_text = self.vectorizer.transform([cleaned_text])
                if debug:
                    logger.info(f"Text Vectorized Successfully. Shape: {vectorized_text.shape}")
                    logger.info(f"Non-Zero Features: {vectorized_text.nnz}")
                
            except Exception as e:
                logger.error(f"Error VEctorizer text: {str(e)}")
                result_dict = {
                    'status': 'error',
                    'error': f"Vectorization Failed: {str(e)}",
                    'prediction': None,
                    'confidence': 0.0,
                    'confidence_level': 'Error',
                    'confidence_interpretation': 'Error',
                    'is_reliable': False,
                    'all_probabilities': {}
                }
                return result_dict if return_dict else PredictionResult(result_dict)
            
            # Convert to Dense array for models that require it
            vectorized_array = vectorized_text.toarray()

            # Get Prediction Probabilities
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(vectorized_array)[0]
                logger.info("Used predict_proba for confidence calculation")
            else:
                # For models without predict_proba, use decision function
                logger.info("Using Decision Function for confidence calculation")
                decision_scores = self.model.decision_function(vectorized_array)[0]
                # Convert to Probabilities using softmax
                exp_scores = np.exp(decision_scores - np.max(decision_scores))
                probabilities = exp_scores / np.sum(exp_scores)
            
            # Make Prediction
            prediction = self.model.predict(vectorized_array)[0]

            # Enhanced Confidence Metrics
            max_confidence = np.max(probabilities)
            second_max_confidence = np.partition(probabilities, -2)[-2]
            confidence_gap = max_confidence - second_max_confidence

            # Entropy-based confidence
            entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
            max_entropy = np.log(len(probabilities))
            normalized_confidence = 1 - (entropy / max_entropy)

            # Get all category probabilities
            all_categories = self.label_encoder.classes_
            category_probabilities = {
                category: float(prob)
                for category, prob in zip(all_categories, probabilities)
            }

            # Sort by probability
            sorted_prediction = sorted(
                category_probabilities.items(),
                key=lambda x: x[1],
                reverse=True
            )

            # Convert prediction back to category name
            predict_category = self.label_encoder.inverse_transform([prediction])[0]

            # Enhanced Confidence Interpretation
            confidence_interpretation = self.get_confidence_interpretation(max_confidence, confidence_gap)

            # Determine if prediction is reliable
            is_reliable = max_confidence > 0.3 and confidence_gap > 0.1

            result_dict = {
                'status': 'success',
                'prediction': predict_category,
                'confidence': float(max_confidence),
                'confidence_level': confidence_interpretation,
                'confidence_gap': float(confidence_gap),
                'normalized_confidence': float(normalized_confidence),
                'entropy': float(entropy),
                'is_reliable': is_reliable,
                'confidence_interpretation': confidence_interpretation,
                'confidence_percentage': f"{max_confidence * 100:.2f}%",
                'top_5_predictions': sorted_prediction[:5],
                'all_probabilities': category_probabilities,
                'model_info': {
                    'model_name': self.metadata.get('best_model_name', 'Unknown'),
                    'model_accuracy': self.metadata.get('accuracy', 'Unknown')
                },
                'debug_info': {
                    'original_text_length': len(resume_text),
                    'cleaned_text_length': len(cleaned_text),
                    'vectorized_features': vectorized_text.nnz,
                    'total_features': vectorized_text.shape[1]
                } if debug else None
            }
            logger.info(f"Enhanced Prediction Completed: {predict_category} with {max_confidence:.4f} confidence")
            logger.info(f"Confidence Gap: {confidence_gap:.4f}, Reliable: {is_reliable}")
            return result_dict if return_dict else PredictionResult(result_dict)
        
        except Exception as e:
            logger.error(f"Error in enhanced prediction: {str(e)}")
            error_result = {
                'status': 'error',
                'error': str(e),
                'prediction': None,
                'confidence': 0.0,
                'confidence_level': 'Error',
                'confidence_interpretation': 'Error',
                'is_raliable': False,
                'all_probabilities': {}
            }
            return error_result if return_dict else PredictionResult(error_result)
        
    
    def prdict_with_confidence(self, resume_text):
        """Original Method - Calls Enhanced Version for backword compatibilty"""
        return self.predict_with_enhanced_confidence(resume_text, debug=False, return_dict=True)
    
    def predict_with_debugging(self, resume_text):
        """Prediction with Full Debugging Information"""
        return self.predict_with_enhanced_confidence(resume_text, debug=True, return_dict=True)
    
    def analyze_prediction_quality(self, resume_text):
        """Analyze Why Confidence Might be Low"""
        try:
            # Get Prediction with Debug Info
            result = self.predict_with_debugging(resume_text)

            if result.get('status') == 'error':
                return result
            
            # Analyze potential issues
            issues = []
            recommendations = []

            # Check Text Length
            if result.get('debug_info', {}).get('cleaned_text_length', 0) < 50:
                issues.append("Very Few Features Extracted from Text")
                recommendations.append("Include more relevant keywords and skills")

            # Check Feature Extraction
            if result.get('debug_info', {}).get('vectorized_faetures', 0) < 10:
                issues.append("Very Few Features Extracted from Text")
                recommendations.append("Include more relevant keywords and skills")

            # Check Confidence Metrics
            if result.get('confidence', 0) < 0.3:
                issues.append("Low Prediction Confidence")
                if result.get('confidence_gap', 0) < 0.1:
                    issues.append("Small Gap between top Prediction")
                    recommendations.append("Resume may span multiple categories")
            
            # Check Entropy
            if result.get('entropy', 0) > 2.0:
                issues.append("High Prediction uncertainty")
                recommendations.append("Resume content may be too generic")
            
            analysis = {
                **result,
                'quality_analysis': {
                    'issues_found': issues,
                    'recommendations': recommendations,
                    'overall_quality': 'Good' if len(issues) == 0 else 'Fair' if len(issues) <= 3 else 'Poor'
                }
            }

            return analysis

        except Exception as e:
            logger.error(f"Error in Prediction quality analysis: {str(e)}")
            return {
                'status': 'error',
                'error': str(e),
                'prediction': None,
                'confidence': 0.0,
                'confidence_level': 'Error'
            }
    
    def predict(self, resume_text):
        """Simple Prediction Method for Backword Compatibility"""
        try:
            result = self.prdict_with_confidence(resume_text)
            if result.get('status') == 'error':
                return None
            return result.get('prediction', None)
        except Exception as e:
            logger.error(f"Error in Simple Prediction: {str(e)}")
            return None
        
    def get_model_info(self):
        """Get Information about the Loaded Model"""
        try:
            info = {
                'model_type': str(type(self.model)),
                'vectorizer_type': str(type(self.vectorizer)),
                'vocabulary_size': len(self.vectorizer.vocabulary_) if hasattr(self.vectorizer, 'vocabulary_') else 0,
                'label_classes': list(self.label_encoder.classes_) if hasattr(self.label_encoder, 'classes_') else [],
                'metadata': self.metadata
            }
            return info
        except Exception as e:
            logger.error(f"Error getting model info: {str(e)}")
            return {}
    
    def validate_pipeline(self):
        """Validate that the Pipeline is Working Correctly"""
        try:
            logger.info("Validating Prediction Pipeline...")

            # Test With Sample Text
            test_text = "Python Programming Machine Learning Data Science Development"
            result = self.prdict_with_confidence(test_text)

            if result.get('status') == 'error':
                logger.error(f"Pipeline Validation Failed: {result.get('error')}")
                return False
            
            logger.info("Pipeline Validation Successfully")
            return True
        
        except Exception as e:
            logger.error(f"Pipeline Validation Failed: {str(e)}")
            return False
        
    def batch_predict(self, resume_texts):
        """Predict Multiple Resume at Once"""
        try:
            results = []
            for i, text in enumerate(resume_texts):
                logger.info(f"Processing Resume {i+1}/{len(resume_texts)}")
                result = self.prdict_with_confidence(text)
                results.append(result)
            return results
        except Exception as e:
            logger.error(f"Error in Batch Prediction: {str(e)}")
            return []
    
    def get_prediction_statistics(self, resume_texts):
        """Get Statistics for a Batch of Predictions"""
        try:
            results = self.batch_predict(resume_texts)

            if not results:
                return {}
            
            # Extract Confidence Score
            confidences = [r.get('confidence', 0) for r in results if r.get('status') == 'success']
            predictions = [r.get('prediction') for r in results if r.get('status') == 'success']

            if not confidences:
                return {'error': 'No successful predictions'}
            
            # Calculate Statistics
            stats = {
                'total_predictions': len(results),
                'successful_predictions': len(confidences),
                'failed_predictions': len(results) - len(confidences),
                'average_confidence': np.mean(confidences),
                'median_confidence': np.median(confidences),
                'min_confidence': np.min(confidences),
                'max_confidence': np.max(confidences),
                'std_confidence': np.std(confidences),
                'prediction_distribution': pd.Series(predictions).value_counts().to_dict(),
                'high_confidence_count': sum(1 for c in confidences if c > 0.7),
                'medium_confidence_count': sum(1 for c in confidences if 0.4 <= c <= 0.7),
                'low_confidence_count': sum(1 for c in confidences if c < 0.4)
            }

            return stats
        except Exception as e:
            logger.error(f"Error Calculationg Prediction Statistics: {str(e)}")
            return {'error': str(e)}


class PredictionResult:
    """Result class to provide both directory and attribute access"""

    def __init__(self, result_dict):
        for key, value in result_dict.items():
            setattr(self, key, value)

        # Add Aliases for Backword Compatibility
        if not hasattr(self, 'confidence_level'):
            self.confidence_level = getattr(self, 'confidence_interpretation', 'Unknown')
        
    def to_dict(self):
        """Convert Back to Dictionary"""
        return {key: value for key, value in self.__dict__.items()}
    
    def get(self, key, default=None):
        """Dictionary-Style get method"""
        return getattr(self, key, default)