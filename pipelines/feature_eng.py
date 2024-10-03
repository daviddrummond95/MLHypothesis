import pandas as pd
import numpy as np
import yaml
import re
import logging
from typing import List, Dict, Any, Tuple

def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def to_numeric_safe(data: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(data[col], errors='coerce')

def fill_empty_values(data: pd.DataFrame, feature: str, feature_groups: Dict[str, List[str]]) -> pd.Series:
    if feature in feature_groups.get('measurements', []):
        return data[feature].fillna(data[feature].median())
    elif feature in feature_groups.get('interventions', []):
        return data[feature].fillna(0)
    elif feature in feature_groups.get('events', []):
        return data[feature].fillna(False if data[feature].dtype == 'bool' else 'No event')
    elif data[feature].dtype in ['int64', 'float64']:
        return data[feature].fillna(data[feature].median())
    else:
        return data[feature].fillna(data[feature].mode()[0] if not data[feature].mode().empty else 'Unknown')

def create_feature(data: pd.DataFrame, new_feature: str, formula: str) -> pd.Series:
    if any(agg in formula for agg in ['max()', 'min()', 'mean()']):
        return evaluate_complex_expression(data, formula)
    else:
        columns = [col.strip() for col in re.split(r'[+\-*/]', formula) if col.strip() in data.columns]
        if not columns:
            logging.warning(f"No valid columns found for feature {new_feature}. Skipping.")
            return pd.Series(index=data.index)
        data[columns] = data[columns].apply(to_numeric_safe)
        return data.eval(formula)

def evaluate_complex_expression(data: pd.DataFrame, expression: str) -> pd.Series:
    def replace_agg(match):
        col, agg = match.groups()
        if col not in data.columns:
            return '0'
        numeric_data = to_numeric_safe(data, col)
        return str(getattr(numeric_data, agg)())

    expression = re.sub(r'(\w+)\.(max|min|mean)\(\)', replace_agg, expression)
    columns = [col.strip() for col in re.split(r'[+\-*/()]', expression) if col.strip() in data.columns]
    data[columns] = data[columns].apply(to_numeric_safe)
    return data.eval(expression)

def create_temporal_features(data: pd.DataFrame, datetime_col: str, group_col: str) -> pd.DataFrame:
    data[datetime_col] = pd.to_datetime(data[datetime_col], errors='coerce')
    data['hour'] = data[datetime_col].dt.hour
    data['day'] = data[datetime_col].dt.day
    data['month'] = data[datetime_col].dt.month
    data['year'] = data[datetime_col].dt.year
    data['dayofweek'] = data[datetime_col].dt.dayofweek
    data['time_since_last'] = data.groupby(group_col)[datetime_col].diff().dt.total_seconds() / 3600
    return data

def encode_categorical_features(data: pd.DataFrame, columns_to_encode: List[str]) -> pd.DataFrame:
    for col in columns_to_encode:
        if col in data.columns:
            data = pd.get_dummies(data, columns=[col], prefix=col, drop_first=True)
        else:
            logging.warning(f"Column {col} not found in the dataset. Skipping encoding for this column.")
    return data

def bin_features(data: pd.DataFrame, bin_config: List[Dict[str, Any]]) -> pd.DataFrame:
    for feature in bin_config:
        feature_name = feature['feature']
        if feature_name not in data.columns:
            logging.warning(f"Feature '{feature_name}' not found in the dataset. Skipping binning for this feature.")
            continue
        
        try:
            bins = [float(x) if x != 'inf' else float('inf') for x in feature['bins']]
            if not all(bins[i] <= bins[i+1] for i in range(len(bins)-1)):
                raise ValueError("Bins are not monotonically increasing")
            
            data[feature_name] = to_numeric_safe(data, feature_name)
            if mapping := feature.get('mapping'):
                data[feature_name] = data[feature_name].map(mapping).fillna(data[feature_name])
            
            data[f"{feature_name}_binned"] = pd.cut(
                data[feature_name],
                bins=bins,
                labels=feature['labels'],
                include_lowest=True
            )
        except Exception as e:
            logging.error(f"Error binning feature '{feature_name}': {str(e)}. Skipping binning for this feature.")
    
    return data

def normalize_features(data: pd.DataFrame, features_to_normalize: List[str]) -> pd.DataFrame:
    for feature in features_to_normalize:
        if feature in data.columns:
            data[feature] = (data[feature] - data[feature].mean()) / data[feature].std()
        else:
            logging.warning(f"Feature '{feature}' not found in the dataset. Skipping normalization for this feature.")
    return data

def create_time_series_features(data: pd.DataFrame, datetime_col: str, group_col: str, value_cols: List[str], config: Dict[str, Any]) -> pd.DataFrame:
    data[datetime_col] = pd.to_datetime(data[datetime_col])
    data = data.sort_values([group_col, datetime_col])

    numeric_cols = [col for col in value_cols if col in data.columns and pd.api.types.is_numeric_dtype(data[col])]
    if not numeric_cols:
        logging.warning("No numeric columns found in value_cols. Skipping time series feature creation.")
        return data

    for col in numeric_cols:
        if config.get('create_lags', False):
            for lag in config.get('lag_values', [1, 2, 3]):
                new_col_name = f'{col}_lag_{lag}'
                data[new_col_name] = data.groupby(group_col)[col].shift(lag)
                logging.info(f"Created lag feature: {new_col_name}")
        
        if config.get('create_diffs', False):
            new_col_name = f'{col}_diff'
            data[new_col_name] = data.groupby(group_col)[col].diff()
            logging.info(f"Created diff feature: {new_col_name}")
        
        if config.get('create_rolling', False):
            for window in config.get('rolling_windows', [3, 5, 7]):
                grouped = data.groupby(group_col)[col].rolling(window=window, min_periods=1)
                data[f'{col}_rolling_mean_{window}'] = grouped.mean().reset_index(level=0, drop=True)
                data[f'{col}_rolling_std_{window}'] = grouped.std().reset_index(level=0, drop=True)
                logging.info(f"Created rolling features for {col} with window {window}")

    return data

def process_features(data: pd.DataFrame, config: Dict[str, Any]) -> Tuple[pd.DataFrame, List[str]]:
    logging.info("Starting feature engineering")

    datetime_col = config.get('datetime_column', 'DateTime')
    group_col = config.get('group_column', 'PatientID')
    
    if datetime_col not in data.columns or group_col not in data.columns:
        raise ValueError(f"Required columns not found in the data: {datetime_col}, {group_col}")
    
    data[datetime_col] = pd.to_datetime(data[datetime_col], errors='coerce')
    
    # Create temporal features
    for feature in config.get('temporal_features', []):
        if feature['type'] == 'time_since_start':
            data[feature['name']] = (data[datetime_col] - data[datetime_col].min()).dt.total_seconds() / 3600
        elif feature['type'] == 'hour_of_day':
            data[feature['name']] = data[datetime_col].dt.hour
        elif feature['type'] == 'day_of_week':
            data[feature['name']] = data[datetime_col].dt.dayofweek
        elif feature['type'] == 'is_weekend':
            data[feature['name']] = data[datetime_col].dt.dayofweek.isin([5, 6]).astype(int)
    
    # Engineer features
    data = engineer_features(data, config)
    
    # Encode categorical variables
    data = encode_categorical_features(data, config.get('categorical_columns', []))

    # Normalize features
    data = normalize_features(data, config.get('normalize_features', []))
    
    # Bin features
    data = bin_features(data, config.get('bin_features', []))
    
    # Create time series features
    if time_series_config := config.get('time_series_features'):
        original_numeric_cols = data.select_dtypes(include=['number']).columns.tolist()[:20]  # Limit to top 10 numeric columns
        data = create_time_series_features(data, datetime_col, group_col, original_numeric_cols, time_series_config)
    
    # Exclude specified features
    data = data.drop(columns=config.get('exclude_features', []), errors='ignore')
    
    # Ensure all columns are numeric and fill NaN values
    feature_groups = config.get('feature_groups', {})
    columns_to_process = [col for col in data.columns if col not in [datetime_col, group_col, 'outcome']]
    
    for col in columns_to_process:
        if col in data.columns:
            try:
                data[col] = to_numeric_safe(data, col)
                data[col] = fill_empty_values(data, col, feature_groups)
            except Exception as e:
                logging.warning(f"Error processing column {col}: {str(e)}. Skipping this column.")
                data = data.drop(columns=[col])

    features = [col for col in data.columns if col not in [datetime_col, group_col, 'outcome']]
    
    return data, features

def engineer_features(data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    logging.info("Starting feature engineering")
    
    features_to_engineer = config.get('engineer_features', {})
    feature_groups = config.get('feature_groups', {})
    
    for new_feature, formula in features_to_engineer.items():
        logging.info(f"Engineering feature: {new_feature}")
        if isinstance(formula, list):
            existing_columns = [col for col in formula if col in data.columns]
            if not existing_columns:
                logging.warning(f"No columns found for feature {new_feature}. Skipping.")
                continue
            data[new_feature] = data[existing_columns].sum(axis=1)
        elif isinstance(formula, str):
            data[new_feature] = create_feature(data, new_feature, formula)
        else:
            logging.warning(f"Unsupported formula type for feature {new_feature}: {type(formula)}")
        
        # Fill NaN values for the new feature
        data[new_feature] = fill_empty_values(data, new_feature, feature_groups)
        logging.info(f"Filled empty values in feature: {new_feature}")
    
    return data

def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    if 'DateTime' in data.columns:
        data['DateTime'] = pd.to_datetime(data['DateTime'], errors='coerce')
    
    if 'Value' in data.columns:
        data['Value'] = to_numeric_safe(data, 'Value')
    
    for col in data.columns:
        data[col] = fill_empty_values(data, col, {})
    
    for col in data.columns:
        if data[col].dtype == 'object':
            data[col] = to_numeric_safe(data, col)
    
    return data