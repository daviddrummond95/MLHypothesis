import matplotlib.pyplot as plt
import seaborn as sns

def interpret_and_report_results(hypothesis_results, causal_results, ml_results):
    report = "Hypothesis Testing Results:\n"
    for feature, result in hypothesis_results.items():
        report += f"{feature}: {result}\n"
    
    report += "\nCausal Inference Results:\n"
    report += f"Estimated Treatment Effect: {causal_results['treatment_effect']}\n"
    report += f"Difference-in-Differences Estimate: {causal_results['did_estimate']}\n"
    
    report += "\nMachine Learning Validation Results:\n"
    report += f"Mean Cross-Validation Score: {ml_results['mean_cv_score']}\n"
    report += "Feature Importances for Hypothesized Features:\n"
    for feature, importance in ml_results['features_of_interest'].items():
        report += f"{feature}: {importance}\n"
    
    # Visualize feature importances
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(ml_results['features_of_interest'].values()), y=list(ml_results['features_of_interest'].keys()))
    plt.title('Feature Importances for Hypothesized Features')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig('feature_importances.png')
    
    return report

# # Usage
# report = interpret_and_report_results(results, 
#                                       {'treatment_effect': treatment_effect, 'did_estimate': did_estimate}, 
#                                       ml_results)
# print(report)