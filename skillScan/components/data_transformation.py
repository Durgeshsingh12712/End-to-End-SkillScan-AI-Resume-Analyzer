import os, sys, re
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

from skillScan.entity import DataTransformationConfig, DataTransformationArtifact
from skillScan.utils import save_object
from skillScan.exception import SkillScanException
from skillScan.logging import logger


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def clean_resume(self, resume_text):
        """Enhanced resume text cleaning"""
        if pd.isna(resume_text):
            return ""
    
        resume_text = str(resume_text).lower()
    
        # Remove URLs
        resume_text = re.sub(r'http\S+|www\S+|https\S+', ' ', resume_text)
    
        # Remove email addresses
        resume_text = re.sub(r'\S+@\S+', ' ', resume_text)
    
        # Remove phone numbers
        resume_text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', ' ', resume_text)
    
        # Remove extra whitespace and special characters
        resume_text = re.sub(r'[^\w\s]', ' ', resume_text)
        resume_text = re.sub(r'\s+', ' ', resume_text)
    
        # Remove very short words (less than 2 characters)
        resume_text = ' '.join([word for word in resume_text.split() if len(word) > 2])
    
        return resume_text.strip()

    def balance_data(self, df, strategy='smart_oversample'):
        """Enhanced data balancing"""
        logger.info(f"Balancing dataset using strategy: {strategy}")
    
        if strategy == 'smart_oversample':
            # Don't oversample to the maximum
            class_counts = df['Category'].value_counts()
            target_size = int(class_counts.quantile(0.75))
        
            balanced_dfs = []
            for category in df['Category'].unique():
                category_df = df[df['Category'] == category]
            
                if len(category_df) < target_size:
                    oversampled = category_df.sample(target_size, replace=True, random_state=42)
                    balanced_dfs.append(oversampled)
                else:
                    balanced_dfs.append(category_df.sample(target_size, random_state=42))
        
            balanced_df = pd.concat(balanced_dfs, ignore_index=True)
        
        elif strategy == 'undersample':
            # Undersample to minimum class size
            min_size = df['Category'].value_counts().min()
            balanced_df = df.groupby('Category').sample(min_size, random_state=42).reset_index(drop=True)
    
        else:  # Original method
            max_size = df['Category'].value_counts().max()
            balanced_df = df.groupby('Category').apply(lambda x: x.sample(max_size, replace=True)).reset_index(drop=True)
    
        balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
        return balanced_df


    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logger.info("Starting data transformation")
            
            # Load data
            df = pd.read_csv(self.config.data_path, encoding='utf-8', on_bad_lines='skip')
            logger.info(f"Data loaded with shape: {df.shape}")
            
            # Remove duplicates
            df = df.drop_duplicates().reset_index(drop=True)
            logger.info(f"After removing duplicates: {df.shape}")
            
            # Clean resume text
            df['cleaned_resume'] = df['Resume'].apply(lambda x: self.clean_resume(x))
            
            # Balance the dataset
            df = self.balance_data(df)
            logger.info(f"After balancing: {df.shape}")

            
            # Label encoding
            label_encoder = LabelEncoder()
            df['Category_encoded'] = label_encoder.fit_transform(df['Category'])
            
            # TF-IDF Vectorization
            tfidf_vectorizer = TfidfVectorizer(
                stop_words='english',
                max_features=self.config.max_features,
                min_df=2,
                max_df=0.95,
                ngram_range=(1, 2),
                lowercase=True,
                strip_accents='unicode',
                sublinear_tf=True,
                norm='l1'
            )
            
            # Fit and transform the cleaned resume text
            X = tfidf_vectorizer.fit_transform(df['cleaned_resume'])
            y = df['Category_encoded']
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            print(X_train, X_test, y_train, y_test)
            # Convert to dense arrays
            X_train_dense = X_train.toarray()
            X_test_dense = X_test.toarray()
            
            # Create column names for features
            feature_names = [f'feature_{i}' for i in range(X_train_dense.shape[1])]
            
            # Create train and test DataFrames
            train_df = pd.DataFrame(X_train_dense, columns=feature_names)
            train_df['target'] = y_train.values
            
            test_df = pd.DataFrame(X_test_dense, columns=feature_names)
            test_df['target'] = y_test.values
            
            # Save train and test data
            train_data_path = os.path.join(self.config.root_dir, "train.csv")
            test_data_path = os.path.join(self.config.root_dir, "test.csv")
            
            train_df.to_csv(train_data_path, index=False)
            test_df.to_csv(test_data_path, index=False)
            
            # Save vectorizer and label encoder
            vectorizer_path = os.path.join(self.config.root_dir, "vectorizer.pkl")
            label_encoder_path = os.path.join(self.config.root_dir, "label_encoder.pkl")
            
            save_object(Path(vectorizer_path), tfidf_vectorizer)
            save_object(Path(label_encoder_path), label_encoder)
            
            logger.info("Data transformation completed successfully")
            
            return DataTransformationArtifact(
                train_data_path=Path(train_data_path),
                test_data_path=Path(test_data_path),
                vectorizer_path=Path(vectorizer_path),
                label_encoder_path=Path(label_encoder_path)
            )
            
        except Exception as e:
            raise SkillScanException(e, sys)

