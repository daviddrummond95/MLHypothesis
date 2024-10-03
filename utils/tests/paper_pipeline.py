from utils.hypothesis.save_hypothesis import get_dataframe, get_all_hypotheses, save_hypothesis_results, get_hypothesis_results
from utils.hypothesis.save_causals import get_causal_results
from utils.hypothesis.test_hypothesis import test_multiple_hypotheses
from utils.hypothesis.save_causals import save_causal_results
from models.causal import perform_linear_regression, perform_random_forest, perform_causal_inference
from pipelines.causal import generate_full_report, format_report, generate_hypothesis_report
from utils.hypothesis.save_causals import save_report_to_database

import logging
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(db_path):
    """Load train and test data from the database."""
    try:
        train_data = get_dataframe(db_path, "train_data")
        test_data = get_dataframe(db_path, "test_data")
        logger.info(f"Loaded train data shape: {train_data.shape}, test data shape: {test_data.shape}")
        logger.info(f"Train data columns: {train_data.columns.tolist()}")
        logger.info(f"Test data columns: {test_data.columns.tolist()}")
        return train_data, test_data
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def prepare_data(data, outcome_column):
    """Prepare data for analysis, handling missing values and encoding."""
    try:
        X = data.drop(columns=[outcome_column])
        y = data[outcome_column]
        
        logger.info(f"Shape before preparation - X: {X.shape}, y: {y.shape}")
        logger.info(f"Columns before preparation: {X.columns.tolist()}")
        
        # # Handle missing values
        # X = X.fillna(X.mean())
        
        # Encode categorical variables
        X = pd.get_dummies(X, drop_first=True)
        
        logger.info(f"Shape after preparation - X: {X.shape}, y: {y.shape}")
        logger.info(f"Columns after preparation: {X.columns.tolist()}")
        
        return X, y
    except Exception as e:
        logger.error(f"Error preparing data: {str(e)}")
        raise




def main():
    try:
        db_path = "hypotheses.db"
        outcome_column = "outcome"
        
        # Load data
        train_data, test_data = load_data(db_path)
        
        # Prepare data
        X_train, y_train = prepare_data(train_data, outcome_column)
        X_test, y_test = prepare_data(test_data, outcome_column)
        
        # Retrieve hypotheses and filter for approved ones
        hypotheses = get_all_hypotheses(db_path)
        approved_hypotheses = [h for h in hypotheses if h.get('validation_result') == 'Approved']
        
        hypothesis_features = []
        hypothesis_types = []
        for h in approved_hypotheses:
            if h['relevant_features']:
                hypothesis_features.append(h['relevant_features'][0])
                hypothesis_types.append('continuous')  # Assuming all are continuous, adjust if needed
        
        logger.info(f"Testing {len(hypothesis_features)} approved hypotheses")
        
        # Test hypotheses
        results = test_multiple_hypotheses(X_train, y_train, hypothesis_features, hypothesis_types)
        logger.info("Hypothesis testing results:")
        for feature, result in results.items():
            logger.info(f"{feature}: {result}")
            # Save hypothesis testing results
            save_hypothesis_results(db_path, feature, 'hypothesis_test', result)
        
        # Perform linear regression
        lr_coefficients = perform_linear_regression(X_train, y_train)
        logger.info("Top 5 linear regression coefficients:")
        logger.info(lr_coefficients.sort_values(key=abs, ascending=False).head().to_dict())
        # Save linear regression results
        for feature, coefficient in lr_coefficients.items():
            save_hypothesis_results(db_path, feature, 'linear_regression', coefficient)
        
        # Perform random forest
        rf_importances = perform_random_forest(X_train, y_train)
        logger.info("Top 5 random forest feature importances:")
        logger.info(rf_importances.sort_values(ascending=False).head().to_dict())
        # Save random forest results
        for feature, importance in rf_importances.items():
            save_hypothesis_results(db_path, feature, 'random_forest', importance)
        
        # Perform causal inference for top features and save results
        top_features = rf_importances.sort_values(ascending=False).head().index
        for feature in top_features:
            try:
                causal_results = perform_causal_inference(X_train, y_train, feature, outcome_column)
                logger.info(f"Causal inference results for {feature}:")
                logger.info(causal_results)
                
                # Save causal results to database
                save_causal_results(db_path, feature, causal_results)
            except Exception as e:
                logger.error(f"Error performing causal inference for feature {feature}: {str(e)}")
                logger.info(f"Skipping causal inference for feature {feature}")
        
        # Retrieve and log all results
        logger.info("Pipeline Results:")
        
        hypothesis_results = get_hypothesis_results(db_path, None)
        logger.info("Hypothesis Testing Results:")
        logger.info(hypothesis_results.get('hypothesis_test', 'N/A'))
        
        logger.info("Linear Regression Coefficient:")
        logger.info(hypothesis_results.get('linear_regression', 'N/A'))
        
        logger.info("Random Forest Feature Importance:")
        logger.info(hypothesis_results.get('random_forest', 'N/A'))
        
        causal_results = get_causal_results(db_path)
        logger.info("Causal Inference Results:")
        logger.info(causal_results if causal_results else 'N/A')
        
        logger.info("Enhanced paper pipeline execution complete.")
    except Exception as e:
        logger.error(f"An error occurred in the main execution: {str(e)}")

    # Define paths to domain and data context files
    domain_files = ['data/Diabetes-Data/README-DIABETES', 'data/Diabetes-Data/Domain-Description']
    data_context_files = ['data/Diabetes-Data/Data-Codes']

    # After you have processed the data and have the pipeline results
    for hypothesis in approved_hypotheses:
        hypothesis_data = {
            "statement": hypothesis['statement'],
            "rationale": hypothesis['rationale'],
            "relevant_features": hypothesis['relevant_features'],
            "expected_effect": hypothesis['expected_effect'],
            "confidence_level": hypothesis['confidence_level']
        }
        pipeline_results = {
            "hypothesis_test": hypothesis_results.get('hypothesis_test', 'N/A'),
            "linear_regression": hypothesis_results.get('linear_regression', 'N/A'),
            "random_forest": hypothesis_results.get('random_forest', 'N/A'),
            "causal_inference": causal_results.get(hypothesis['relevant_features'][0], 'N/A')
        }
        report = generate_hypothesis_report(hypothesis_data, pipeline_results, domain_files, data_context_files)
        if report:
            # Save the report to the database
            save_report_to_database(db_path, hypothesis['id'], report)
            logging.info(f"Report for hypothesis {hypothesis['id']} generated and saved to database.")
        else:
            logging.warning(f"Failed to generate report for hypothesis {hypothesis['id']}.")

if __name__ == "__main__":
    main()