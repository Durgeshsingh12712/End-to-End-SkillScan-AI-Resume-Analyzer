import os, uuid, tempfile

from datetime import datetime
from werkzeug.utils import secure_filename
from flask import Flask, request, render_template, jsonify, send_file, flash, redirect, url_for, session

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

from skillScan.logging import logger
from skillScan.utils import extract_text_from_file
from skillScan.pipeline import PredictionPipeline

# Configure
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['MAX_CONTENT_LENGHT'] = 2 * 1024 * 1024 # 2MB Max
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()

# Initialize ML Pipeline
try:
    ml_pipeline = PredictionPipeline()
    logger.info("ML Pipeline Initialized Successfully")
except Exception as e:
    logger.error(f"Error to Initialized ML Pipeline: {str(e)}")
    ml_pipeline = None

# Allowd File Extensions
AllOWED_EXTENSIONS = {'pdf', 'docx'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in AllOWED_EXTENSIONS

def get_confidence_level_from_score(confidence_score):
    """Convert Confidence score to readable level"""
    if confidence_score > 0.8:
        return "Very High"
    elif confidence_score > 0.6:
        return "High"
    elif confidence_score > 0.4:
        return "Medium"
    elif confidence_score > 0.25:
        return "Low"
    else:
        return "Very Low"


def validate_and_fix_prediction_result(prediction_result):
    """Validate and fix Prediction result structure"""

    # Log the Raw Prediction Result for Debugging
    logger.debug(f"Raw Prediction Result Type: {type(prediction_result)}")
    logger.debug(f"Raw Prediction Result: {prediction_result}")

    # Handle PredictionResult Object (Convert to Dict)
    if hasattr(prediction_result, 'to_dict'):
        prediction_result = prediction_result.to_dict()
        logger.info("Converted PredictionResult Object to Dictionary")
    
    # Handle Non-Dictionary Results
    if not isinstance(prediction_result, dict):
        logger.warning(f"Prediction Result is not a Dictionary (type: {type(prediction_result)}), Creating Default Structure")

        # Try to extract usefull Information from Non-dict Results
        if isinstance(prediction_result, str):
            # If it's a string, assume it's the prediction
            return {
                'prediction': prediction_result,
                'confidence': 0.0,
                'confidence_level': 'Very Low',
                'top_5_predictions': [],
                'status': 'success'
            }
        elif isinstance(prediction_result, (list, tuple)) and len(prediction_result) > 0:
            # If it's a string, assume it's the prediction
            confidence_val = float(prediction_result[1]) if len(prediction_result) > 1 and isinstance(prediction_result[1], (int, float)) else 0.0
            return {
                'prediction': str(prediction_result[0]),
                'confidence': confidence_val,
                'confidence_level': get_confidence_level_from_score(confidence_val),
                'top_5_predictions': [],
                'status': 'success'
            }
        else:
            # Default FallBack
            return {
                'prediction': 'Unknown',
                'confidence': 0.0,
                'confidence_level': 'Very Low',
                'top_5_predictions': [],
                'status': 'success'
            }
    
    # Ensure All Required Fields Exist with Proper Defaults
    required_fields = {
        'prediction': 'Unknown',
        'confidence': 0.0,
        'confidence_level': 'Very Low',
        'top_5_predictions': [],
        'status': 'success'
    }

    for field, default_value in required_fields.items():
        if field not in prediction_result:
            logger.warning(f"Missing Field '{field}' in prediction result, using default")
            prediction_result[field] = default_value
    
    # Validate and fix Confidence Value
    if not isinstance(prediction_result['confidence'], (int, float)):
        logger.warning(f"Invalid confidence value type: {type(prediction_result['confidence'])}, setting to 0.0")
        prediction_result['confidence'] = 0.0
    
    # Ensure Confidence is between 0 and 1
    if prediction_result['confidence'] > 1:
        prediction_result['confidence'] = prediction_result['confidence'] / 100
    
    # Fix confidence_level mapping
    if 'confidence_ineterpretation' in prediction_result and prediction_result['confidence_level'] in ['Very Low', 'Low', 'Medium', 'High', 'Very High']:
        pass
    elif 'confidence_interpretation' in prediction_result:
        # Map Confidence_interpretation to confidence_level
        interpretation = prediction_result['confidence_interpretation']
        if 'Very High' in interpretation:
            prediction_result['confidence_level'] = 'Very High'
        elif 'High' in interpretation:
            prediction_result['confidence_level'] = 'High'
        elif 'Medium' in interpretation:
            prediction_result['confidence_level'] = 'Medium'
        elif 'Low' in interpretation and 'Very Low' not in interpretation:
            prediction_result['confidence_level'] = 'Low'
        else:
            prediction_result['confidence_level'] = 'Very Low'
    else:
        # Generate confidence_level from confidence score
        prediction_result['confidence_level'] = get_confidence_level_from_score(prediction_result['confidence'])
    
    # Validate top_5_prediction structure
    if not isinstance(prediction_result['top_5_predictions'], list):
        logger.warning("Invalid top_5_predictions structure, setting to empty list")
        prediction_result['top_5_predictions'] = []
    
    # Ensure top_5_predictions has the right format
    fixed_predictions = []
    for item in prediction_result['top_5_predictions']:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            fixed_predictions.append([str(item[0]), float(item[1])])
        elif isinstance(item, dict) and 'category' in item and 'score' in item:
            fixed_predictions.append([str(item['category']), float(item['score'])])
        else:
            logger.warning(f"Invalid prediction item format: {item}")
    
    prediction_result['top_5_predictions'] = fixed_predictions

    # Ensure we Have a proper prediction
    if not prediction_result['prediction'] or prediction_result['prediction'] == 'Unknown':
        if fixed_predictions:
            prediction_result['prediction'] = fixed_predictions[0][0]
            logger.info(f"Set Prediction from Top Prediction: {prediction_result['prediction']}")
    
    logger.info(f"Final Prediction Result: {prediction_result['prediction']} with confidence {prediction_result['confidence']} ({prediction_result['confidence_level']})")

    return prediction_result

def analyze_skill_and_suggestion(prediction_result, resume_text):
    """Generate skill Analysis and Improvement Suggestions"""

    # Extract key skills from resume text (basic keyword matching)
    skill_keywords = [
        # Technical Skills
        'python', 'java', 'javascript', 'c++', 'c#', 'sql', 'html', 'css', 'react', 'angular',
        'node.js', 'php', 'ruby', 'swift', 'kotlin', 'scala', 'go', 'rust', 'typescript',
        'machine learning', 'data science', 'artificial intelligence', 'deep learning',
        'computer vision', 'natural language processing','generative ai','big data','MLOps'
        'tensorflow', 'pytorch', 'scikit-learn', 'pandas', 'numpy', 'matplotlib',
        'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'git', 'github',
        'agile', 'scrum', 'devops', 'ci/cd', 'microservices', 'api', 'rest', 'graphql',
        
        # Soft Skills
        'leadership', 'communication', 'teamwork', 'problem solving', 'analytical',
        'project management', 'time management', 'critical thinking', 'creativity',
        'adaptability', 'collaboration', 'presentation', 'negotiation'
    ]

    resume_lower = resume_text.lower()
    found_skills = []

    for skill in skill_keywords:
        if skill.lower() in resume_lower:
            found_skills.append(skill.title())

    # Generate category-specific suggestions
    predicted_category = prediction_result.get('prediction', 'Unknown')
    confidence = prediction_result.get('confidence', 0)

    category_suggestions = {
        'Data Science': [
            'Consider adding experience with statistical analysis tools (R, SPSS)',
            'Highlight data visualization projects (Tableau, Power BI)',
            'Include machine learning model deployment experience',
            'Add cloud platform experience (AWS SageMaker, Azure ML)',
            'Showcase big data technologies (Spark, Hadoop)'
        ],
        'Web Development': [
            'Add modern JavaScript frameworks (React, Vue.js, Angular)',
            'Include responsive design and mobile-first development',
            'Highlight API development and integration experience',
            'Add database management skills (MongoDB, PostgreSQL)',
            'Include version control and collaboration tools'
        ],
        'Software Engineering': [
            'Add system design and architecture experience',
            'Include testing methodologies (unit, integration, TDD)',
            'Highlight performance optimization projects',
            'Add containerization and orchestration skills',
            'Include security best practices and implementation'
        ],
        'DevOps': [
            'Add infrastructure as code experience (Terraform, CloudFormation)',
            'Include monitoring and logging tools (Prometheus, ELK Stack)',
            'Highlight automation and scripting skills',
            'Add security and compliance experience',
            'Include disaster recovery and backup strategies'
        ]
    }

    # General suggestion based on confidence level
    general_suggestions = []
    if confidence < 0.5:
        general_suggestions.extend([
            'Consider adding more specific technical skills and tools',
            'Include quantifiable achievements and project outcomes',
            'Add relevant certifications and training',
            'Highlight industry-specific keywords and terminology'
        ])
    
    # Get category-specific suggestions
    specific_suggestions = category_suggestions.get(predicted_category, [
        'Add more relevant technical skills for your target role',
        'Include specific tools and technologies you have used',
        'Highlight project outcomes and measurable results',
        'Consider adding relevant certifications'
    ])

    return {
        'found_skills': found_skills,
        'skill_count': len(found_skills),
        'improvement_suggestions': general_suggestions + specific_suggestions[:3],
        'category_focus': predicted_category,
        'confidence_level': prediction_result.get('confidence_level', 'Unknown')
    }

def cleanup_temp_file(file_path):
    """Clean up temporary file"""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Cleaned up Temporary files: {file_path}")
    except Exception as e:
        logger.error(f"Error cleaning up file {file_path}: {str(e)}")

@app.route('/')
def index():
    """Main Upload Page"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle File Upload and Processing"""
    if ml_pipeline is None:
        flash('ML Pipeline not Available.Please Try again later.', 'error')
        return redirect(url_for('index'))
    
    # Check If File was Uploaded
    if 'file' not in request.files:
        flash('No File Selected', 'error')
        return redirect(url_for('index'))
    
    file = request.files['file']
    if file.filename == '':
        flash('No File Selected', 'error')
        return redirect(url_for('index'))
    
    if file and allowed_file(file.filename):
        file_path = None
        try:
            # Create Secure Filename and Save Temporarily
            filename = secure_filename(file.filename)
            file_extension = filename.rsplit('.', 1)[1].lower()
            temp_filename = f"{uuid.uuid4()}_{filename}"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)

            file.save(file_path)
            logger.info(f"File Saved Temporarily: {file_path}")

            # Extract Text and File
            resume_text = extract_text_from_file(file_path, file_extension)

            if not resume_text:
                cleanup_temp_file(file_path)
                flash('Could not extract text from file. Please check file format.', 'error')
                return redirect(url_for('index'))
            
            if len(resume_text.strip()) < 50:
                cleanup_temp_file(file_path)
                flash('Resume content is too short for analysis. Please provide a more detailed resume.', 'error')
                return redirect(url_for('index'))
            
            # Analyze with ML Pipeline - wth Enhanced error handling
            try:
                logger.info(f"Starting ML Analysis for files: {filename}")
                raw_prediction_result = ml_pipeline.predict_with_enhanced_confidence(resume_text, debug=True)
                logger.info(f"Ml Analysis Completed for files: {filename}")
                logger.debug(f"Raw ML Result  Type: {type(raw_prediction_result)}")
                logger.debug(f"Raw ML Result: {raw_prediction_result}")

                prediction_result = validate_and_fix_prediction_result(raw_prediction_result)
                logger.info(f"Processed Prediction Result: {prediction_result}")
            except Exception as e:
                logger.error(f"ML Pipeline Error for {filename}: {str(e)}")
                logger.erro(f"ML Error Type: {type(e).__name__}")

                # Create a fallback result
                prediction_result = {
                    'prediction': 'Analysis Error',
                    'confidence': 0.0,
                    'confidence_level': 'Very Low',
                    'top_5_predictions': [],
                    'status': 'error',
                    'error': str(e)
                }

            # Check If Analysis Failed
            if prediction_result.get('status') == 'error':
                cleanup_temp_file(file_path)
                error_msg = prediction_result.get('error', 'Unknown error occured during analysis')
                flash(f"Analysis Failed: {error_msg}", 'error')
                return redirect(url_for('index'))
                
            # Analyze skills and generate suggestions
            try:
                skill_analysis = analyze_skill_and_suggestion(prediction_result, resume_text)
            except Exception as e:
                logger.error(f"Skill Analysis Error: {str(e)}")
                skill_analysis = {
                    'found_skills': [],
                    'skill_count': 0,
                    'improvement_suggestions': ['Unable to analyze skills due to processing error'],
                    'category_focus': 'Unknown',
                    'confidence_level': 'Unknown'
                }
                
            # Prepare Results for Display
            results = {
                'filename': filename,
                'upload_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'prediction': prediction_result,
                'skill_analysis': skill_analysis,
                'resume_length': len(resume_text),
                'resume_preview': resume_text[:500] + '...' if len(resume_text) > 500 else resume_text
            }

            # Clean up Temporary File
            cleanup_temp_file(file_path)

            # Log Final Results for Debugging
            logger.info(f"Final Result for {filename}")
            logger.info(f"  - Prediction: {results['prediction']['prediction']}")
            logger.info(f"  - Confidence: {results['prediction']['confidence']}")
            logger.info(f"  - Confidence Level: {results['prediction']['confidence_level']}")
            logger.info(f"  - Top Predictions: {len(results['prediction']['top_5_predictions'])}")

            # Try to render template with error handling
            try:
                return render_template('results.html', results=results)
            except Exception as e:
                logger.error(f"Template rendering error: {str(e)}")
                flash(f"Error Displaying Results: {str(e)}", 'error')
                return redirect(url_for('index'))
                
        except Exception as e:
            # Clean up on Error
            if file_path and os.path.exists(file_path):
                cleanup_temp_file(file_path)
                
            logger.error(f"Error Processing File {filename}: {str(e)}")
            logger.error(f"Error Details: {type(e).__name__}: {str(e)}")
            flash(f"Error Processing File: {str(e)}", 'error')
            return redirect(url_for('index'))
            
    else:
        flash('Invalid File Type. PLease Upload PDF or DOCX Files Only')
        return redirect(url_for('index'))

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """API Endpoint for Resume Analysis"""
    if ml_pipeline is None:
        return jsonify({'error': 'ML Pipiline not Available'}), 500
    
    try:
        #Check if File was Uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No File Uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '' or not allowed_file(file.filename):
            return jsonify({'error': 'Invalid File'}), 400
        
        # Process File
        filename = secure_filename(file.filename)
        file_extension = filename.rsplit('.', 1)[1].lower()
        temp_filename = f"{uuid.uuid4()}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)

        file.save(file_path)

        # Extract Text
        resume_text = extract_text_from_file(file_path, file_extension)

        if not resume_text:
            cleanup_temp_file(file_path)
            return jsonify({'error': 'Could not extract text from file'}), 400
        
        # Analyze
        raw_prediction_result = ml_pipeline.predict_with_enhanced_confidence(resume_text)
        prediction_result = validate_and_fix_prediction_result(raw_prediction_result)
        skill_analysis = analyze_skill_and_suggestion(prediction_result, resume_text)

        # Cleanup
        cleanup_temp_file(file_path)

        return jsonify({
            'status': 'success',
            'prediction': prediction_result,
            'skill_analysis': skill_analysis,
            'filename': filename,
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.error(f"API Error: {str(e)}")
        return jsonify({'error': {str(e)}}), 500


@app.route('/download-report')
def download_report():
    """Generate and download PDF analysis report"""
    try:
        # Create temporary PDF file
        report_path = os.path.join(tempfile.gettempdir(), f'resume_analysis_{uuid.uuid4()}.pdf')
        
        # Create PDF
        c = canvas.Canvas(report_path, pagesize=letter)
        width, height = letter
        
        # Add content to PDF
        c.setFont("Helvetica-Bold", 16)
        c.drawString(50, height - 50, "Resume Analysis Report")
        
        c.setFont("Helvetica", 12)
        c.drawString(50, height - 80, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Add your report content here
        y_position = height - 120
        c.drawString(50, y_position, "Analysis Results:")
        
        # Add more content as needed
        y_position -= 30
        c.drawString(50, y_position, "• Skills identified")
        y_position -= 20
        c.drawString(50, y_position, "• Experience summary")
        y_position -= 20
        c.drawString(50, y_position, "• Recommendations")
        
        c.save()
        
        # Send file for download
        return send_file(
            report_path, 
            as_attachment=True, 
            download_name='resume_analysis.pdf',
            mimetype='application/pdf'
        )
        
    except Exception as e:
        logger.error(f"Error generating PDF report: {str(e)}")
        flash('Error generating report', 'error')
        return redirect(url_for('index'))

@app.errorhandler(413)
def too_large(e):
    flash('File too large. Maximum size is 2MB.', 'error')
    return redirect(url_for('index'))

@app.errorhandler(404)
def not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(e):
    return render_template('500.html'), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)