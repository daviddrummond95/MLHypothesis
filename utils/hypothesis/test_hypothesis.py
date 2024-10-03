from scipy import stats
import statsmodels.api as sm
import pandas as pd
import numpy as np
import logging
try:
    from statsmodels.stats.multicomp import multipletests
except ImportError:
    # If multipletests is not available, we'll implement a simple Bonferroni correction
    def multipletests(pvals, alpha=0.05, method='bonferroni'):
        n = len(pvals)
        corrected_pvals = np.minimum(pvals * n, 1.0)
        reject = corrected_pvals < alpha
        return reject, corrected_pvals, None, None

def test_hypothesis(X, y, feature, hypothesis_type='continuous'):
    if hypothesis_type == 'continuous':
        try:
            # Perform t-test
            success_group = X[feature][y == 1]
            failure_group = X[feature][y == 0]
            t_stat, p_value = stats.ttest_ind(success_group, failure_group)
            return f"T-statistic: {t_stat}, p-value: {p_value}"
        except Exception as e:
            logging.error(f"Error performing t-test for feature {feature}: {e}")
            logging.info(f"X Columns: {X.columns}")
            return f"Error performing t-test for feature {feature}: {e}"
    elif hypothesis_type == 'categorical':
        # Perform chi-square test
        contingency_table = pd.crosstab(X[feature], y)
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
        return f"Chi-square statistic: {chi2}, p-value: {p_value}"
    elif hypothesis_type == 'multivariate':
        # Perform logistic regression
        X_with_const = sm.add_constant(X[feature])
        model = sm.Logit(y, X_with_const).fit()
        return model.summary()

def test_multiple_hypotheses(X, y, features, hypothesis_types):
    results = {}
    p_values = []
    for feature, h_type in zip(features, hypothesis_types):
        result = test_hypothesis(X, y, feature, h_type)
        results[feature] = result
        if h_type != 'multivariate':
            p_values.append(float(result.split('p-value: ')[1]))
    
    # Correct for multiple testing
    rejected, corrected_p_values, _, _ = multipletests(p_values, method='fdr_bh')
    
    for i, feature in enumerate(results):
        if hypothesis_types[i] != 'multivariate':
            results[feature] += f", Corrected p-value: {corrected_p_values[i]}"
    
    return results

# Usage
# hypotheses = {
#     'cd4_count': 'continuous',
#     'age': 'continuous',
#     'treatment_history': 'categorical',
#     'viral_load, cd4_count': 'multivariate'
# }

# results = test_multiple_hypotheses(X_train, y_train, list(hypotheses.keys()), list(hypotheses.values()))
# for feature, result in results.items():
#     print(f"Hypothesis test for {feature}:")
#     print(result)
#     print()