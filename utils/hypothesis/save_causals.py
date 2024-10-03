import sqlite3
import json
import logging

logger = logging.getLogger(__name__)

def save_causal_results(db_path, feature, results):
    """
    Save causal inference results to the database.
    
    :param db_path: Path to the SQLite database
    :param feature: The feature for which causal inference was performed
    :param results: Dictionary containing causal inference results
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create table if it doesn't exist
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS causal_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                feature TEXT NOT NULL,
                results TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Insert results
        cursor.execute('''
            INSERT INTO causal_results (feature, results)
            VALUES (?, ?)
        ''', (feature, json.dumps(results)))
        
        conn.commit()
        logger.info(f"Causal results for feature '{feature}' saved to database.")
    except Exception as e:
        logger.error(f"Error saving causal results to database: {str(e)}")
        raise
    finally:
        conn.close()

def get_causal_results(db_path, feature=None):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    if feature:
        cursor.execute("SELECT feature, results FROM causal_results WHERE feature = ? ORDER BY timestamp DESC LIMIT 1", (feature,))
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return {result[0]: json.loads(result[1])}
    else:
        cursor.execute("SELECT feature, results FROM causal_results GROUP BY feature HAVING MAX(timestamp)")
        results = cursor.fetchall()
        conn.close()
        
        if results:
            return {row[0]: json.loads(row[1]) for row in results}
    
    return {}


def save_report_to_database(db_path, hypothesis_id, report):
    """Save the generated report to the database."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create table if it doesn't exist
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS generated_reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                hypothesis_id TEXT NOT NULL,
                report TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Insert report
        cursor.execute('''
            INSERT INTO generated_reports (hypothesis_id, report)
            VALUES (?, ?)
        ''', (hypothesis_id, report))
        
        conn.commit()
        logger.info(f"Report for hypothesis {hypothesis_id} saved to database.")
    except Exception as e:
        logger.error(f"Error saving report to database: {str(e)}")
        raise
    finally:
        conn.close()

def get_report_from_database(db_path, hypothesis_id):
    """Retrieve the generated report for a specific hypothesis from the database."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT report FROM generated_reports
            WHERE hypothesis_id = ?
            ORDER BY timestamp DESC
            LIMIT 1
        ''', (hypothesis_id,))
        
        result = cursor.fetchone()
        
        if result:
            return result[0]
        else:
            return None
    except Exception as e:
        logger.error(f"Error retrieving report from database: {str(e)}")
        return None
    finally:
        conn.close()