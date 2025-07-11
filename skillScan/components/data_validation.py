import os, sys
from pathlib import Path

from skillScan.logging import logger
from skillScan.exception import SkillScanException
from skillScan.entity import DataValidationConfig, DataValidationArtifact


class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    def validate_all_files_exist(self) -> bool:
        """Validate if all required files exist in the unzipped data directory"""
        try:
            validation_status = True
            missing_files = []

            # Check if the unzip directory exists
            unzip_data_dir = Path(self.config.root_dir).parent / "data_ingestion"

            if not unzip_data_dir.exists():
                logger.error(f"Unzip directory does not exists: {unzip_data_dir}")
                validation_status = False
                with open(self.config.STATUS_FILE, 'w') as f:
                    f.write(f"Validation Status: {validation_status}\nError: Unzip directory not found")
                return validation_status
            
            # Get all files in the unzip directory
            all_files  = os.listdir(unzip_data_dir)
            logger.info(f"Files Found in {unzip_data_dir}: {all_files}")

            # Check if all required files are present
            for required_file in self.config.ALL_REQUIRED_FILES:
                if required_file not in all_files:
                    missing_files.append(required_file)
                    validation_status = False
            
            # Write Validation Status to file
            with open(self.config.STATUS_FILE, 'w') as f:
                if validation_status:
                    f.write(f"Validation Status: {validation_status}\nAll required files are present")
                    logger.info("All Required Files are Present")
                else:
                    f.write(f"Validation Status; {validation_status}\nMissing Files: {missing_files}")
                    logger.info(f"Missing required files: {missing_files}")

            return validation_status
        
        except Exception as e:
            logger.error(f"Error during file validation: {str(e)}")
            validation_status = False
            with open(self.config.STATUS_FILE, 'w') as f:
                f.write(f"Validation Status: {validation_status}\nError: {str(e)}")
            raise SkillScanException(e, sys)
    

    def validation_file_extensions(self) -> bool:
        """Validation file extension are as expected"""
        try:
            unzip_data_dir = Path(self.config.root_dir).parent / "data_ingestion"

            if not unzip_data_dir.exists():
                return False
            
            all_files = os.listdir(unzip_data_dir)

            #Define expected extension (you can modify this based on your needs)
            excepted_extensions = ['.csv', '.json', '.txt', '.xlsx', '.zip']

            for file in all_files:
                file_ext = Path(file).suffix.lower()
                if file_ext not in excepted_extensions:
                    logger.warning(f"Unexpected file extension: {file}")
                
            return True
        
        except Exception as e:
            logger.error(f"Error during file extension validation: {str(e)}")
            return False
            
    
    def validation_file_size(self) -> bool:
        """Validation that files are not empty and within reasonale size limits"""
        try:
            unzip_data_dir = Path(self.config.root_dir).parent / "data_ingestion"

            if not unzip_data_dir.exists():
                return False
            
            all_files = os.listdir(unzip_data_dir)

            for file in all_files:
                file_path = unzip_data_dir / file
                if file_path.is_file():
                    file_size = file_path.stat().st_size

                    # Check if file is empty
                    if file_size == 0:
                        logger.error(f"Empty file detected: {file}")
                        return False
                    
                    # Check if file is too large
                    max_size = 500 * 1024 * 1024
                    if file_size > max_size:
                        logger.warning(f"Large file detected: {file}({file_size / (1024*1024):.2f} MB)")

            return True
        
        except Exception as e:
            logger.error(f"Error during file size validation: {str(e)}")
            return False
    
    def initiate_data_validation(self) -> DataValidationArtifact:
        """Main method to initiate comprehensive Data Validation"""
        try:
            logger.info("Data Validation Started")

            # Create the status file directory if it doesn't exist
            status_file_dir = Path(self.config.STATUS_FILE).parent
            status_file_dir.mkdir(parents=True, exist_ok= True)

            # Perform all validations
            files_exist = self.validate_all_files_exist()
            extensions_valid = self.validation_file_extensions()
            sizes_valid = self.validation_file_size()

            # Overall validation status
            validation_status = files_exist and extensions_valid and sizes_valid

            logger.info(f"Files Exist Validation: {files_exist}")
            logger.info(f"Extenstions Validation: {extensions_valid}")
            logger.info(f"file Size Validation: {sizes_valid}")
            logger.info(f"Overall Data Validation Status: {validation_status}")

            # Update status file with comprehensive results
            with open(self.config.STATUS_FILE, 'a') as f:
                f.write(f"\nFile Extension Validation: {extensions_valid}")
                f.write(f"\nFile Size Validation: {sizes_valid}")
                f.write(f"\nOverall Validation status: {validation_status}")
            
            return DataValidationArtifact(validation_status=validation_status)

        except Exception as e:
            logger.error(f"Error during Data Validation Initiation: {str(e)}")
            raise SkillScanException(e, sys)

print("✅ Data Validation component implemented!")