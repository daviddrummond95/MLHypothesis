from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from statsmodels.stats.multitest import multipletests
from causalinference import CausalModel
from dowhy import CausalModel as DoWhyCausalModel
from econml.dml import CausalForestDML
import logging
import pandas as pd
import numpy as np
from scipy import stats

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_hypothesis(X, y, feature, hypothesis_type='continuous'):
    """Test a single hypothesis."""
    try:
        if hypothesis_type == 'continuous':
            correlation, p_value = stats.pearsonr(X[feature], y)
            return {
                'correlation': correlation,
                'p_value': p_value,
                'significant': p_value < 0.05
            }
        elif hypothesis_type == 'categorical':
            groups = [y[X[feature] == val] for val in X[feature].unique()]
            f_statistic, p_value = stats.f_oneway(*groups)
            return {
                'f_statistic': f_statistic,
                'p_value': p_value,
                'significant': p_value < 0.05
            }
        else:
            raise ValueError(f"Unsupported hypothesis type: {hypothesis_type}")
    except Exception as e:
        logger.error(f"Error testing hypothesis for feature {feature}: {str(e)}")
        raise


def test_multiple_hypotheses(X, y, hypotheses):
    """Test multiple hypotheses with correction for multiple testing."""
    results = {}
    p_values = []
    
    for feature, h_type in hypotheses:
        try:
            result = test_hypothesis(X, y, feature, h_type)
            results[feature] = result
            p_values.append(result['p_value'])
        except Exception as e:
            logger.error(f"Error testing hypothesis for feature {feature}: {str(e)}")
    
    # Correct for multiple testing
    rejected, corrected_p_values, _, _ = multipletests(p_values, method='fdr_bh')
    
    for (feature, _), corrected_p in zip(hypotheses, corrected_p_values):
        if feature in results:
            results[feature]['corrected_p_value'] = corrected_p
            results[feature]['significant_after_correction'] = corrected_p < 0.05
    
    return results

def perform_linear_regression(X, y):
    """Perform linear regression and return coefficients."""
    try:
        model = LinearRegression()
        model.fit(X, y)
        return pd.Series(model.coef_, index=X.columns)
    except Exception as e:
        logger.error(f"Error performing linear regression: {str(e)}")
        raise

def perform_random_forest(X, y):
    """Perform random forest and return feature importances."""
    try:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        return pd.Series(model.feature_importances_, index=X.columns)
    except Exception as e:
        logger.error(f"Error performing random forest: {str(e)}")
        raise

def perform_causal_inference(X, y, treatment, outcome):
    """Perform causal inference using various methods."""
    results = {}
    
    try:
        # Prepare data
        data = X.copy()
        data[outcome] = y
        
        # CausalInference package
        cm = CausalModel(
            Y=data[outcome].values,
            D=data[treatment].values,
            X=data.drop(columns=[treatment, outcome]).values
        )
        cm.est_via_ols()
        results['causalinference_ate'] = cm.estimates['ate']
        
        # DoWhy package
        model = DoWhyCausalModel(
            data=data,
            treatment=treatment,
            outcome=outcome,
            common_causes=data.columns.drop([treatment, outcome]).tolist()
        )
        identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
        estimate = model.estimate_effect(identified_estimand, method_name="backdoor.linear_regression")
        results['dowhy_ate'] = estimate.value
        
        # EconML package
        cf = CausalForestDML(n_estimators=100, random_state=42)
        cf.fit(Y=data[outcome], T=data[treatment], X=data.drop(columns=[treatment, outcome]))
        te_pred = cf.effect(data.drop(columns=[treatment, outcome]))
        results['econml_ate'] = np.mean(te_pred)
        
        return results
    except Exception as e:
        logger.error(f"Error performing causal inference for treatment {treatment}: {str(e)}")
        return results