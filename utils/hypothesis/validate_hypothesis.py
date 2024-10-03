import sqlite3
import pandas as pd
from scipy import stats
import logging

def validate_hypothesis(db_path, hypothesis_id, validation_data, validation_result=None, supporting_data=None):
    """
    Validate a hypothesis and update the database with the validation results.
    
    Args:
        db_path (str): Path to the SQLite database file.
        hypothesis_id (int): ID of the hypothesis to validate.
        validation_data (pd.DataFrame): The validation dataset.
        validation_result (str, optional): The validation result ("Approved" or "Declined").
        supporting_data (str, optional): The reason for approval or an empty string for decline.
    
    Returns:
        None
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM hypotheses WHERE id = ?', (hypothesis_id,))
    hypothesis = cursor.fetchone()
    
    if hypothesis is None:
        logging.warning(f"Hypothesis with id {hypothesis_id} not found")
        return
    
    # If validation_result and supporting_data are provided, update the hypothesis
    if validation_result is not None and supporting_data is not None:
        cursor.execute("""
            UPDATE hypotheses
            SET validation_result = ?,
                supporting_data = ?
            WHERE id = ?
        """, (validation_result, supporting_data, hypothesis_id))
        
        conn.commit()
        print(f"Hypothesis {hypothesis_id} validated. Result: {validation_result}")
    else:
        print(f"Hypothesis {hypothesis_id} ready for validation.")
    
    conn.close()
