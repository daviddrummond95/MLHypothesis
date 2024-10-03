import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

def preprocess_data(data, config):
    date_column = config['date_column']
    time_column = config['time_column']
    code_column = config['code_column']
    value_column = config['value_column']
    
    data[date_column] = pd.to_datetime(data[date_column], format=config['date_format'], errors='coerce')
    data[time_column] = data[time_column].apply(convert_time)
    data['DateTime'] = pd.to_datetime(
        data[date_column].dt.strftime('%Y-%m-%d') + ' ' + data[time_column].astype(str),
        format='%Y-%m-%d %H:%M:%S',
        errors='coerce'
    )
    
    data = data.sort_values(['PatientID', 'DateTime'])
    
    # Create columns for each code
    for code, column_name in config['measurement_codes'].items():
        data[column_name] = np.where(data[code_column] == int(code), data[value_column], np.nan)
        data[column_name] = pd.to_numeric(data[column_name], errors='coerce')
    
    data = data.drop(columns=[date_column, time_column, code_column, value_column])
    
    # Log the columns after preprocessing
    logging.info(f"Columns after preprocessing: {data.columns.tolist()}")
    
    return data

def prepare_target_variable(data, config):
    glucose_columns = config.get('glucose_measurement_columns', [])
    if not glucose_columns:
        raise ValueError("No glucose measurement columns specified in the configuration")

    # Log the columns in the data
    logging.info(f"Columns in the data: {data.columns.tolist()}")
    
    # Log the glucose columns we're looking for
    logging.info(f"Glucose columns specified in config: {glucose_columns}")

    measurement_column = next((col for col in glucose_columns if col in data.columns), None)
    if measurement_column is None:
        # Log the columns that are missing
        missing_columns = [col for col in glucose_columns if col not in data.columns]
        logging.error(f"Missing glucose measurement columns: {missing_columns}")
        raise ValueError("No matching glucose measurement column found in the data")

    threshold = config.get('outcome_threshold', 140)
    
    data[measurement_column] = pd.to_numeric(data[measurement_column], errors='coerce')
    data['next_measurement'] = data.groupby('PatientID')[measurement_column].shift(-1)
    data['next_measurement'] = pd.to_numeric(data['next_measurement'], errors='coerce')
    data['outcome'] = (data['next_measurement'] > threshold).astype(int)
    data = data.dropna(subset=['outcome'])
    
    return data

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