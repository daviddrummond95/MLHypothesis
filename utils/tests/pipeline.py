import pandas as pd
import logging
import requests
import zipfile
import tarfile
import io
import os
import yaml
import numpy as np
from sklearn.model_selection import train_test_split
from datetime import datetime, time, timedelta
from pipelines.data_prep import preprocess_data, prepare_target_variable
from pipelines.feature_eng import process_features, create_time_series_features
from models.predictive import train_predictive_model, evaluate_model, calculate_feature_importance, preprocess_features
from models.morphing import morph_profile, enforce_constraints, calculate_similarity
from models.synth_data_gen import train_vae, generate_synthetic_data
from utils.hypothesis.save_hypothesis import create_hypotheses_table, save_hypothesis, get_all_hypotheses, save_pipeline_results, save_dataframe, get_dataframe
from utils.hypothesis.validate_hypothesis import validate_hypothesis
from pipelines.hypothesis_gen import generate_and_analyze_hypotheses, prepare_hypotheses_for_saving
import os 
from sklearn.dummy import DummyClassifier

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load configurations
with open('configs/preprocess.yaml', 'r') as file:
    preprocess_config = yaml.safe_load(file)

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

# Step 1: Download and load the dataset
logging.info("Step 1: Downloading and loading the dataset")
url = "https://archive.ics.uci.edu/static/public/34/diabetes.zip"
data_dir = "data"
diabetes_data_dir = os.path.join(data_dir, "Diabetes-Data")

preprocess_config = load_config('configs/preprocess.yaml')
feature_config = load_config('configs/features.yaml')

def download_and_extract_data(url, data_dir):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    response = requests.get(url)
    if response.status_code == 200:
        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
            zip_ref.extractall(data_dir)
            logging.info(f"Files in the zip archive: {zip_ref.namelist()}")
        logging.info(f"Data extracted to {data_dir}")
    else:
        raise Exception(f"Failed to download data: HTTP {response.status_code}")

    # Extract the tar file
    tar_file = os.path.join(data_dir, 'diabetes-data.tar')
    if os.path.exists(tar_file):
        with tarfile.open(tar_file, 'r') as tar:
            tar.extractall(path=data_dir)
        logging.info(f"Extracted contents of {tar_file}")
    else:
        logging.error(f"Tar file not found: {tar_file}")

download_and_extract_data(url, data_dir)

# Log the contents of the Diabetes-Data directory
logging.info(f"Files in the Diabetes-Data directory: {os.listdir(diabetes_data_dir)}")

# Look for data files that start with "data-"
data_files = [f for f in os.listdir(diabetes_data_dir) if f.startswith('data-')]
if not data_files:
    raise Exception("No suitable data file found in the extracted data")

logging.info(f"Potential data files found: {data_files}")

def convert_time(time_str):
    if pd.isna(time_str):
        return None
    
    time_str = str(time_str).strip()
    
    if ':' in time_str:
        try:
            time = datetime.strptime(time_str, '%H:%M')
        except ValueError:
            # Handle cases where hours might be > 24
            hours, minutes = map(int, time_str.split(':'))
            hours = hours % 24  # Ensure hours are within 0-23 range
            time = datetime(1, 1, 1) + timedelta(hours=hours, minutes=minutes)
    else:
        try:
            time = datetime.strptime(time_str, '%H%M')
        except ValueError:
            # If it's a single number, assume it's hours
            hours = int(time_str) % 24  # Ensure hours are within 0-23 range
            time = datetime(1, 1, 1) + timedelta(hours=hours)
    
    return time.strftime('%H:%M')

def load_and_preprocess_data(data_dir):
    all_data = []
    for data_file in os.listdir(data_dir):
        if data_file.startswith('data-'):
            file_path = os.path.join(data_dir, data_file)
            df = pd.read_csv(file_path, sep='\t', header=None, names=['Date', 'Time', 'Code', 'Value'])
            patient_id = data_file.split('-')[1].split('.')[0]
            df['PatientID'] = patient_id
            all_data.append(df)
    
    if not all_data:
        raise Exception("Failed to read any data files")
    
    data = pd.concat(all_data, ignore_index=True)
    
    # Use the generic preprocess_data function
    data = preprocess_data(data, preprocess_config)
    
    return data

def log_nan_percentages(df, stage):
    nan_percentages = (df.isna().sum() / len(df)) * 100
    logging.info(f"\n{stage} - Percentage of NaN values per column:")
    for col, pct in nan_percentages.items():
        logging.info(f"{col}: {pct:.2f}%")

def main():
    # Step 1: Download and load the dataset
    logging.info("Step 1: Downloading and loading the dataset")
    url = "https://archive.ics.uci.edu/static/public/34/diabetes.zip"
    data_dir = "data"
    diabetes_data_dir = os.path.join(data_dir, "Diabetes-Data")
    
    if not os.path.exists(diabetes_data_dir):
        os.makedirs(diabetes_data_dir)
        response = requests.get(url)
        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
            zip_ref.extractall(data_dir)
    
    data = load_and_preprocess_data(diabetes_data_dir)
    logging.info(f"Loaded data shape: {data.shape}")
    logging.info(f"Columns in loaded data: {data.columns.tolist()}")
    
    # Log NaN percentages for original data
    log_nan_percentages(data, "Original Data")
    
    # Step 2: Apply time series feature engineering
    logging.info("Step 2: Applying time series feature engineering")
    time_series_features = feature_config.get('time_series_features', {})
    numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
    logging.info(f"Numeric columns for time series features: {numeric_cols}")
    data = create_time_series_features(
        data,
        datetime_col='DateTime',
        group_col='PatientID',
        value_cols=numeric_cols,
        config=time_series_features
    )
    logging.info(f"Data shape after time series feature engineering: {data.shape}")
    logging.info(f"Columns after time series feature engineering: {data.columns.tolist()}")
    
    # Load preprocessing config
    preprocess_config = load_config('configs/preprocess.yaml')

    # Step 3: Process features
    logging.info("Step 3: Processing features")
    data, features = process_features(data, feature_config)
    logging.info(f"Data shape after feature processing: {data.shape}")
    logging.info(f"Processed features: {features}")

    # Step 4: Prepare the target variable
    logging.info("Step 4: Preparing the target variable")
    logging.info(f"Columns before preparing target variable: {data.columns.tolist()}")
    try:
        data = prepare_target_variable(data, preprocess_config)
        logging.info(f"Data shape after preparing target variable: {data.shape}")
        logging.info(f"Columns after preparing target variable: {data.columns.tolist()}")
        
        for col in data.columns:
            logging.info(f"Column '{col}' - dtype: {data[col].dtype}, unique values: {data[col].nunique()}")
            if data[col].dtype == 'object':
                logging.info(f"Sample values for '{col}': {data[col].sample(5).tolist()}")
        
    except ValueError as e:
        logging.error(f"Error in preparing target variable: {str(e)}")
        raise
    
    # Check if 'outcome' is present in the data
    if 'outcome' not in data.columns:
        logging.warning("'outcome' column not found. Skipping steps that require the target variable.")
        return
    
    # Check target variable distribution
    if 'outcome' in data.columns:
        target_distribution = data['outcome'].value_counts(normalize=True)
        logging.info(f"Target variable distribution:\n{target_distribution}")
    else:
        logging.warning("Target variable 'outcome' not found in the data")
    
    # Split the data
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    logging.info(f"Train data shape: {train_data.shape}, Test data shape: {test_data.shape}")
    
    # Step 5: Feature Engineering
    logging.info("Step 5: Feature Engineering")
    train_data, train_features = process_features(train_data, feature_config)
    test_data, test_features = process_features(test_data, feature_config)
    
    # Ensure 'next_glucose' and 'Code' are not in the feature list
    train_features = [f for f in train_features if f not in ['next_measurement', 'Code', 'outcome_diff']]
    test_features = [f for f in test_features if f not in ['next_measurement', 'Code', 'outcome_diff']]
    
    # Ensure both datasets have the same features
    common_features = list(set(train_features) & set(test_features))
    train_data = train_data[common_features + ['outcome']]
    test_data = test_data[common_features + ['outcome']]
    
    logging.info(f"Train data shape after feature engineering: {train_data.shape}")
    logging.info(f"Test data shape after feature engineering: {test_data.shape}")
    logging.info(f"Common features: {common_features}")
    
    # Log NaN percentages after feature engineering
    log_nan_percentages(train_data, "After Feature Engineering (Train Data)")
    log_nan_percentages(test_data, "After Feature Engineering (Test Data)")
    
    # Step 6: Train Predictive Model
    logging.info("Step 6: Training Predictive Model")
    X_train = train_data.drop(columns=['outcome'])
    y_train = train_data['outcome']
    
    X_train = preprocess_features(X_train)
    model, feature_names = train_predictive_model(train_data, target_column="outcome")
    
    # Ensure test data has the same columns as training data
    X_test = test_data.drop(columns=["outcome"])
    X_test = preprocess_features(X_test)
    X_test = X_test.reindex(columns=feature_names, fill_value=-1)
    
    y_test = test_data['outcome']
    
    # Make predictions
    y_pred = model.predict(X_test)
    logging.info(f"Predicted class distribution: {np.unique(y_pred, return_counts=True)}")
    
    # Calculate and log evaluation metrics
    evaluation_metrics = evaluate_model(model, X_test, y_test)
    logging.info(f"Model evaluation metrics: {evaluation_metrics}")
    
    # Calculate feature importance
    feature_importance = calculate_feature_importance(model, X_train)
    logging.info(f"Feature Importance: {feature_importance[:5]}")  # Log top 5 features
    
    # Step 7: Synthetic Data Generation
    logging.info("Step 7: Synthetic Data Generation")
    try:
        vae, encoder, decoder, scaler, columns = train_vae(train_data)
        synthetic_df = generate_synthetic_data(decoder, len(train_data), 10, scaler, columns)
        if synthetic_df is not None:
            logging.info(f"Generated synthetic data shape: {synthetic_df.shape}")
            # Convert one-hot encoded columns back to categorical
            for col in feature_config['feature_groups']['events'] + feature_config['feature_groups']['interventions']:
                if col not in synthetic_df.columns and any(synthetic_df.columns.str.startswith(f"{col}_")):
                    one_hot_cols = [c for c in synthetic_df.columns if c.startswith(f"{col}_")]
                    synthetic_df[col] = synthetic_df[one_hot_cols].idxmax(axis=1).str.replace(f"{col}_", "")
                    synthetic_df = synthetic_df.drop(columns=one_hot_cols)

            # Round numeric columns to appropriate precision
            for col in synthetic_df.select_dtypes(include=['float64']).columns:
                synthetic_df[col] = synthetic_df[col].round(2)

            # Ensure 'PatientID' is present and unique
            if 'PatientID' not in synthetic_df.columns:
                synthetic_df['PatientID'] = range(1, len(synthetic_df) + 1)

            # Log NaN percentages for synthetic data
            log_nan_percentages(synthetic_df, "Synthetic Data")
        else:
            logging.warning("Failed to generate synthetic data")
    except Exception as e:
        logging.error(f"Error in synthetic data generation: {str(e)}")
        synthetic_df = None
    
    if synthetic_df is None:
        logging.warning("Skipping synthetic data processing due to generation error")

    # Step 8: Morphing Procedure
    logging.info("Step 8: Morphing Procedure")
    original_profile = train_data.iloc[0].to_dict()
    morphed_profile = morph_profile(original_profile, direction="increase")
    morphed_profile = enforce_constraints(morphed_profile)
    similarity = calculate_similarity(original_profile, morphed_profile)
    logging.info(f"Similarity between original and morphed profile: {similarity}")
    
    # Step 9: Hypothesis Generation and Analysis
    logging.info("Step 9: Hypothesis Generation and Analysis")
    if synthetic_df is not None:
        target_definition = "The target variable 'outcome' represents the next glucose measurement for a patient."
        hypotheses_with_insights = generate_and_analyze_hypotheses(synthetic_df, train_data, target_definition)
        logging.info(f"Number of hypotheses generated and analyzed: {len(hypotheses_with_insights)}")
        # for i, hypothesis in enumerate(hypotheses_with_insights[:5]):  # Log first 5 hypotheses
        #     logging.info(f"Hypothesis {i}:")
        #     logging.info(f"  Statement: {hypothesis['statement']}")
        #     logging.info(f"  Reasoning: {hypothesis['llm_reasoning']}")
        #     logging.info(f"  Explanation: {hypothesis['explanation']}")
        #     logging.info(f"  Validity: {hypothesis['validity']}")
        
        prepared_hypotheses = prepare_hypotheses_for_saving(hypotheses_with_insights)
    else:
        logging.warning("Skipping hypothesis generation and analysis due to missing synthetic data")
        hypotheses_with_insights = []
        prepared_hypotheses = []
    
    # Step 10: Save Hypotheses to Database
    logging.info("Step 10: Saving Hypotheses to Database")
    db_path = "hypotheses.db"
    create_hypotheses_table(db_path)
    for hypothesis in prepared_hypotheses:
        save_hypothesis(db_path, hypothesis)
    logging.info(f"Hypotheses saved to database: {db_path}")
    
    # Step 11: Validate Hypotheses
    logging.info("Step 11: Validating Hypotheses")
    for hypothesis in get_all_hypotheses(db_path):
        validate_hypothesis(db_path, hypothesis['id'], test_data)
    logging.info("Hypothesis validation complete")
    
    # Save pipeline results
    pipeline_results = {
        'evaluation_metrics': evaluation_metrics,
        'feature_importance': feature_importance,
        'original_profile': original_profile,
        'morphed_profile': morphed_profile,
        'similarity': similarity
    }
    
    save_pipeline_results(db_path, pipeline_results)
    
    logging.info("Pipeline results saved to database")
    
    # Print final results
    logging.info("Pipeline execution complete. Final results:")
    logging.info(f"Evaluation Metrics: {evaluation_metrics}")
    logging.info(f"Top 5 Feature Importance: {feature_importance[:5]}")
    logging.info(f"Original Profile: {original_profile}")
    logging.info(f"Morphed Profile: {morphed_profile}")
    logging.info(f"Similarity: {similarity}")
    
    # Log model information
    logging.info(f"Model type: {type(model).__name__}")
    logging.info(f"Model parameters: {model.get_params()}")
    
    # After feature engineering and before model training
    logging.info("Saving train and test data to database")
    save_dataframe(db_path, train_data, "train_data")
    save_dataframe(db_path, test_data, "test_data")
    logging.info("Train and test data saved to database")
    
    # At the end of the main function, add:
    logging.info("Retrieving saved train and test data from database")
    retrieved_train_data = get_dataframe(db_path, "train_data")
    retrieved_test_data = get_dataframe(db_path, "test_data")
    logging.info(f"Retrieved train data shape: {retrieved_train_data.shape}")
    logging.info(f"Retrieved test data shape: {retrieved_test_data.shape}")

    logging.info("Pipeline execution complete.")

if __name__ == "__main__":
    main()



