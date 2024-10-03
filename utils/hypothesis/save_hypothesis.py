import sqlite3
from typing import Dict
import json
import pandas as pd
import logging
import numpy as np

def create_hypotheses_table(db_path: str):
    """
    Create the hypotheses table in the SQLite database if it doesn't exist.
    
    Args:
        db_path (str): Path to the SQLite database file.
    
    Returns:
        None
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS hypotheses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            statement TEXT NOT NULL,
            rationale TEXT NOT NULL,
            relevant_features TEXT NOT NULL,
            expected_effect TEXT NOT NULL,
            confidence_level REAL NOT NULL,
            validation_result INTEGER,
            supporting_data TEXT,
            llm_reasoning TEXT,
            llm_explanation TEXT,
            llm_validity BOOLEAN
        )
    ''')
    conn.commit()
    conn.close()

def save_hypothesis(db_path: str, hypothesis: Dict):
    """
    Save a hypothesis to the SQLite database.
    
    Args:
        db_path (str): Path to the SQLite database file.
        hypothesis (Dict): Dictionary containing hypothesis details.
    
    Returns:
        None
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        confidence_level = float(hypothesis.get('confidence_level', 0.0))
        if np.isnan(confidence_level):
            logging.warning(f"Skipping hypothesis with NaN confidence level: {hypothesis}")
            return
    except (TypeError, ValueError):
        logging.warning(f"Invalid confidence_level: {hypothesis.get('confidence_level')}")
        return
    
    sql = '''
        INSERT INTO hypotheses (statement, rationale, relevant_features, expected_effect, confidence_level, validation_result, supporting_data, llm_reasoning, llm_explanation, llm_validity)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    '''
    params = (
        hypothesis.get('statement', ''),
        hypothesis.get('rationale', ''),
        ','.join(hypothesis['relevant_features']) if isinstance(hypothesis.get('relevant_features'), list) else hypothesis.get('relevant_features', ''),
        hypothesis.get('expected_effect', ''),
        confidence_level,
        hypothesis.get('validation_result'),
        hypothesis.get('supporting_data'),
        hypothesis.get('llm_reasoning', ''),
        hypothesis.get('llm_explanation', ''),
        hypothesis.get('llm_validity')
    )
    
    logging.info(f"Saving hypothesis: {hypothesis}")
    logging.info(f"Executing SQL: {sql}")
    logging.info(f"With parameters: {params}")
    
    cursor.execute(sql, params)
    
    conn.commit()
    conn.close()

    logging.info(f"Hypothesis saved successfully: {hypothesis.get('statement', '')}")

def get_all_hypotheses(db_path: str):
    """
    Retrieve all hypotheses from the SQLite database.
    
    Args:
        db_path (str): Path to the SQLite database file.
    
    Returns:
        List[Dict]: List of hypotheses.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM hypotheses')
    rows = cursor.fetchall()
    
    hypotheses = []
    for row in rows:
        try:
            hypothesis = {
                'id': row[0],
                'statement': row[1],
                'rationale': row[2],
                'relevant_features': row[3].split(',') if row[3] else [],
                'expected_effect': row[4],
                'confidence_level': row[5],
                'validation_result': row[6],
                'supporting_data': row[7],
                'llm_reasoning': row[8],
                'llm_explanation': row[9],
                'llm_validity': row[10]
            }
            hypotheses.append(hypothesis)
            logging.info(f"Retrieved hypothesis: {hypothesis}")
        except Exception as e:
            logging.error(f"Error processing hypothesis row: {row}")
            logging.error(f"Error details: {str(e)}")
    
    conn.close()
    return hypotheses

def save_pipeline_results(db_path: str, results: Dict):
    """
    Save pipeline results to the SQLite database.
    
    Args:
        db_path (str): Path to the SQLite database file.
        results (Dict): Dictionary containing pipeline results.
    
    Returns:
        None
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS pipeline_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            evaluation_metrics TEXT,
            feature_importance TEXT,
            original_profile TEXT,
            morphed_profile TEXT,
            similarity REAL
        )
    ''')
    
    cursor.execute('''
        INSERT INTO pipeline_results (evaluation_metrics, feature_importance, original_profile, morphed_profile, similarity)
        VALUES (?, ?, ?, ?, ?)
    ''', (
        json.dumps(results['evaluation_metrics']),
        json.dumps(results['feature_importance']),
        json.dumps(results['original_profile'], default=str),
        json.dumps(results['morphed_profile'], default=str),
        results['similarity']
    ))
    
    conn.commit()
    conn.close()

def get_pipeline_results(db_path: str):
    """
    Retrieve pipeline results from the SQLite database.
    
    Args:
        db_path (str): Path to the SQLite database file.
    
    Returns:
        Dict: Pipeline results.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM pipeline_results ORDER BY id DESC LIMIT 1')
    row = cursor.fetchone()
    
    if row is None:
        return None
    
    results = {
        'evaluation_metrics': json.loads(row[1]),
        'feature_importance': json.loads(row[2]),
        'original_profile': json.loads(row[3]),
        'morphed_profile': json.loads(row[4]),
        'similarity': row[5]
    }
    
    conn.close()
    return results

def save_dataframe(db_path: str, df: pd.DataFrame, table_name: str):
    """
    Save a DataFrame to the SQLite database.
    
    Args:
        db_path (str): Path to the SQLite database file.
        df (pd.DataFrame): DataFrame to save.
        table_name (str): Name of the table to save the DataFrame to.
    
    Returns:
        None
    """
    conn = sqlite3.connect(db_path)
    df.to_sql(table_name, conn, if_exists='replace', index=False)
    conn.close()

def get_dataframe(db_path: str, table_name: str) -> pd.DataFrame:
    """
    Retrieve a DataFrame from the SQLite database.
    
    Args:
        db_path (str): Path to the SQLite database file.
        table_name (str): Name of the table to retrieve the DataFrame from.
    
    Returns:
        pd.DataFrame: Retrieved DataFrame.
    """
    conn = sqlite3.connect(db_path)
    df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
    conn.close()
    return df

def get_hypothesis(db_path, hypothesis_id):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM hypotheses WHERE id = ?", (hypothesis_id,))
    hypothesis = cursor.fetchone()
    conn.close()
    if hypothesis:
        return dict(zip([column[0] for column in cursor.description], hypothesis))
    return None

def save_hypothesis_results(db_path, feature, result_type, result):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS hypothesis_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            feature TEXT NOT NULL,
            result_type TEXT NOT NULL,
            result TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    cursor.execute('''
        INSERT INTO hypothesis_results (feature, result_type, result)
        VALUES (?, ?, ?)
    ''', (feature, result_type, json.dumps(result)))
    
    conn.commit()
    conn.close()

def get_hypothesis_results(db_path, feature):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT result_type, result FROM hypothesis_results
        WHERE feature = ?
        GROUP BY result_type
        HAVING MAX(timestamp)
    ''', (feature,))
    
    results = {row[0]: json.loads(row[1]) for row in cursor.fetchall()}
    
    conn.close()
    return results
