import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from models.morphing import morph_profile, enforce_constraints, calculate_similarity
from utils.hypothesis.save_hypothesis import create_hypotheses_table, save_hypothesis, get_all_hypotheses
from utils.hypothesis.validate_hypothesis import validate_hypothesis
from sklearn.utils.class_weight import compute_class_weight
import logging
from sklearn.dummy import DummyClassifier

# Load configuration
with open('configs/pred_model.yaml', 'r') as file:
    config = yaml.safe_load(file)

def train_predictive_model(data, target_column="outcome"):
    # Separate features and target
    X = data.drop(columns=[target_column])
    y = data[target_column]

    X = preprocess_features(X)

    # Initialize and train the model based on config
    model_type = config['model_type']
    if model_type == 'xgboost':
        model = XGBClassifier(**config['xgboost_params'])
    elif model_type == 'random_forest':
        model = RandomForestClassifier(**config['random_forest_params'])
    elif model_type == 'logistic_regression':
        model = LogisticRegression(**config['logistic_regression_params'])
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    model.fit(X, y)

    return model, X.columns  # Return the model and the column names

def preprocess_features(X):
    # Convert 'DateTime' to numerical features if present
    if 'DateTime' in X.columns:
        X['hour'] = X['DateTime'].dt.hour
        X['day'] = X['DateTime'].dt.day
        X['month'] = X['DateTime'].dt.month
        X['year'] = X['DateTime'].dt.year
        X = X.drop(columns=['DateTime'])

    # Force all columns to be numeric
    for col in X.columns:
        if X[col].dtype == 'object':
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
        else:
            X[col] = pd.to_numeric(X[col], errors='coerce')

    # Fill NaN values with -1
    X = X.fillna(-1)

    return X.astype(float)  # Ensure all columns are float

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {}
    for metric in config['evaluation_metrics']:
        if metric == 'accuracy':
            metrics['accuracy'] = accuracy_score(y_test, y_pred)
        elif metric == 'precision':
            metrics['precision'] = precision_score(y_test, y_pred, zero_division=0)
        elif metric == 'recall':
            metrics['recall'] = recall_score(y_test, y_pred, zero_division=0)
        elif metric == 'f1_score':
            metrics['f1'] = f1_score(y_test, y_pred, zero_division=0)
        elif metric == 'roc_auc':
            if len(np.unique(y_test)) > 1:
                metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
            else:
                metrics['roc_auc'] = None
                logging.warning("Only one class present in the target variable. ROC AUC score is not defined.")
    
    return metrics

def calculate_feature_importance(model, X):
    if isinstance(model, DummyClassifier):
        feature_names = X.columns
        importance = np.ones(len(feature_names)) / len(feature_names)
    elif hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
        feature_names = X.columns
    elif hasattr(model, "coef_"):
        importance = np.abs(model.coef_[0])
        feature_names = X.columns
    else:
        raise ValueError("Model does not have feature importances or coefficients.")
    
    feature_importance = [
        {"feature": str(name), "importance": float(imp)}
        for name, imp in zip(feature_names, importance)
    ]
    
    feature_importance.sort(key=lambda x: x["importance"], reverse=True)
    
    return feature_importance

def generate_hypotheses(synthetic_profiles, train_data):
    # Placeholder function for hypothesis generation
    hypotheses = []
    for i in range(5):  # Generate 5 sample hypotheses
        hypothesis = {
            'statement': f"Hypothesis {i+1}",
            'rationale': f"Rationale for hypothesis {i+1}",
            'relevant_features': ['feature_1', 'feature_2'],
            'expected_effect': "Increase in target variable",
            'confidence_level': 0.7
        }
        hypotheses.append(hypothesis)
    return hypotheses

def prioritize_hypotheses(hypotheses):
    # Placeholder function for hypothesis prioritization
    # For now, just sort by confidence level in descending order
    return sorted(hypotheses, key=lambda h: h['confidence_level'], reverse=True)