import pandas as pd
import numpy as np
from typing import List, Dict
from scipy import stats
import logging
import yaml
import re
from langchain_openai import ChatOpenAI
from concurrent.futures import ThreadPoolExecutor

model = ChatOpenAI(model='gpt-4o')

def generate_hypotheses(synthetic_data, real_data):
    """
    Generate hypotheses based on synthetic and real patient profiles.
    
    Args:
        synthetic_data (pd.DataFrame): DataFrame containing synthetic patient profiles.
        real_data (pd.DataFrame): The real dataset.
    
    Returns:
        List[Dict]: List of generated hypotheses.
    """
    # Load exclude_features from YAML file
    with open('configs/hypothesis.yaml', 'r') as file:
        config = yaml.safe_load(file)
    exclude_patterns = config.get('exclude_features', [])

    hypotheses = []
    
    features_to_analyze = [col for col in synthetic_data.columns if col != 'outcome']
    
    # Filter out excluded features
    features_to_analyze = [
        col for col in features_to_analyze
        if not any(re.match(pattern.replace('%', '.*'), col, re.IGNORECASE) for pattern in exclude_patterns)
    ]
    
    for column in features_to_analyze:
        if column in real_data.columns:
            synthetic_values = synthetic_data[column].dropna()
            real_values = real_data[column].dropna()
            
            if len(synthetic_values) == 0 or len(real_values) == 0:
                logging.warning(f"Skipping {column} due to insufficient non-NaN values")
                continue
            
            if pd.api.types.is_numeric_dtype(synthetic_values) and pd.api.types.is_numeric_dtype(real_values):
                synthetic_mean = synthetic_values.mean()
                real_mean = real_values.mean()
                mean_diff = synthetic_mean - real_mean
                
                if np.isnan(mean_diff):
                    logging.warning(f"Skipping {column} due to NaN mean difference")
                    continue
                
                mean_diff_str = f"{mean_diff:.2f}"
                
                # Avoid division by zero
                denominator = max(abs(synthetic_mean), abs(real_mean), 1e-10)
                confidence = min(abs(mean_diff) / denominator, 1.0)
                
                if confidence > 0.1:  # Only generate hypothesis if confidence is above a threshold
                    hypothesis = {
                        "statement": f"The synthetic data shows a different distribution for {column} compared to the real data.",
                        "rationale": f"Observed a mean difference of {mean_diff_str} for {column}.",
                        "relevant_features": [column],
                        "expected_effect": "increase" if mean_diff > 0 else "decrease",
                        "confidence_level": confidence
                    }
                    hypotheses.append(hypothesis)
            
            elif pd.api.types.is_categorical_dtype(synthetic_values) or pd.api.types.is_object_dtype(synthetic_values):
                synthetic_value_counts = synthetic_values.value_counts(normalize=True)
                real_value_counts = real_values.value_counts(normalize=True)
                
                diff = np.sum(np.abs(synthetic_value_counts - real_value_counts.reindex(synthetic_value_counts.index, fill_value=0))) / 2
                
                if diff > 0.1:  # Only generate hypothesis if the difference is significant
                    hypothesis = {
                        "statement": f"The synthetic data shows a different distribution for {column} compared to the real data.",
                        "rationale": f"Observed a distribution difference of {diff:.2f} for {column}.",
                        "relevant_features": [column],
                        "expected_effect": "distribution change",
                        "confidence_level": diff
                    }
                    hypotheses.append(hypothesis)
    
    return hypotheses

def prioritize_hypotheses(hypotheses: List[Dict]) -> List[Dict]:
    """
    Prioritize the generated hypotheses based on predefined criteria.
    
    Args:
        hypotheses (List[Dict]): List of generated hypotheses.
    
    Returns:
        List[Dict]: Prioritized list of hypotheses.
    """
    # Sort hypotheses by confidence score
    prioritized_hypotheses = sorted(hypotheses, key=lambda x: x['confidence_level'], reverse=True)
    return prioritized_hypotheses

def prepare_hypotheses_for_saving(hypotheses: List[Dict]) -> List[Dict]:
    """
    Prepare hypotheses for saving to the database.
    
    Args:
        hypotheses (List[Dict]): List of generated hypotheses.
    
    Returns:
        List[Dict]: List of hypotheses ready for database insertion.
    """
    prepared_hypotheses = []
    for hypothesis in hypotheses:
        try:
            prepared_hypothesis = {
                "statement": hypothesis.get("statement", ""),
                "rationale": hypothesis.get("rationale", ""),
                "relevant_features": hypothesis.get("relevant_features", []),
                "expected_effect": hypothesis.get("expected_effect", ""),
                "confidence_level": float(hypothesis.get("confidence_level", 0.0)),
                "validation_result": hypothesis.get("validation_result"),
                "supporting_data": hypothesis.get("supporting_data"),
                "llm_reasoning": hypothesis.get("llm_reasoning", ""),
                "llm_explanation": hypothesis.get("llm_explanation", ""),
                "llm_validity": 1 if hypothesis.get("llm_validity") else 0
            }
            prepared_hypotheses.append(prepared_hypothesis)
            logging.info(f"Prepared hypothesis for saving: {prepared_hypothesis}")
        except Exception as e:
            logging.warning(f"Error preparing hypothesis for saving: {str(e)}")
            logging.warning(f"Problematic hypothesis: {hypothesis}")
            continue
    
    return prepared_hypotheses

def hypothesis_reasoning_insights(hypotheses: List[Dict], target_definition: str) -> List[Dict]:
    """
    Uses LLM to generate insights from the hypotheses. 

    Feature Metrics -> LLM -> Explanation
    
    Args:
        hypotheses (List[Dict]): List of generated hypotheses.
        target_definition (str): The target definition.
    Returns:
        List[Dict]: List of insights.
    """
    system_prompt = """ You are a scientist evaluating hypotheses from the perspective of plausible causal relationships. You will be provided the following:

    - Feature Metrics: A list of feature metrics for the hypothesis.
    - Hypothesis: A hypothesis statement.
    - Rationale: A rationale for the hypothesis.
    - Relevant Features: A list of relevant features for the hypothesis.
    - Expected Effect: The expected effect of the hypothesis.
    - Confidence Level: The confidence level of the hypothesis.
    - Target Definition: The target definition.
    
    Your task is to generate an explanation for the hypothesis or if it is not valid you may state as such. 

    Please note that valid in this case means PLAUSABLE, not proved. If you can imagine, some causal link to be tested, it is valid.

    Response Format:
    <reasoning>
    Your thought process here. You should be critical of the hypothesis and provide a rationale for why it is valid or not. Present both the positive and negative aspects of the hypothesis.
    The main objective is really coming up with a potential link between the variable and the outcome that would make the hypothesis plausible. 
    </reasoning>
    <explanation>
    The explanation for the hypothesis. Even if the hypothesis is not valid, you should provide an explanation for why it is not valid.
    </explanation>
    <validity>
    True or False
    </validity>

    Two Example Responses

    **Example 1**:
   <reasoning>
    This hypothesis presents a nuanced view of how changes in Regular insulin doses over time affect subsequent glucose measurements. Let's analyze its strengths and potential limitations:

    Positive aspects:
    1. **Time-dependent effects**: The hypothesis correctly acknowledges that the impact of insulin on glucose levels varies over time, aligning with the known pharmacokinetics of Regular insulin.
    2. **Multiple lag consideration**: By including three lag periods, the hypothesis captures both immediate and residual effects of insulin dose changes.
    3. **Graduated impact**: The expected diminishing effect from lag_1 to lag_3 reflects the natural decay of insulin action over time.
    4. **Biological plausibility**: The rationale is grounded in established pharmacological principles of insulin action.

    Negative aspects:
    1. **Simplification of complex dynamics**: While more nuanced than a single-lag model, this hypothesis still simplifies the complex, non-linear relationship between insulin and glucose.
    2. **Individual variability**: The hypothesis doesn't account for potential differences in insulin sensitivity or metabolism among individuals.
    3. **External factors**: The impact of other glucose-influencing factors (e.g., meals, exercise) is not explicitly considered.
    4. **Assumption of consistent dosing intervals**: The hypothesis assumes regular, consistent intervals between insulin doses, which may not always be the case in real-world scenarios.

    The proposed relationship between lagged insulin dose changes and glucose measurements is plausible and well-grounded in pharmacological principles. The use of multiple lag periods allows for a more comprehensive model of insulin's time-dependent effects.
    </reasoning>

    <explanation>
    The hypothesis that changes in Regular insulin doses at different time lags have varying effects on the next glucose measurement, with diminishing impact over time, is valid and well-supported by pharmacological knowledge.

    Regular insulin typically begins to lower blood glucose within 30 minutes of administration, reaches peak effectiveness around 2-3 hours, and can continue to have effects for up to 6 hours. This pharmacokinetic profile aligns well with the proposed lag structure:

    1. dose_diff_lag_1 likely captures the period of peak insulin action, hence the expected strong negative correlation with the next glucose measurement.
    2. dose_diff_lag_2 may represent the latter part of the peak action and the beginning of the decline, explaining the expected moderate negative correlation.
    3. dose_diff_lag_3 likely falls within the tail end of insulin action, accounting for the expected weak negative correlation.

    This graduated effect across different time lags provides a more sophisticated model of glucose dynamics compared to single-lag models. It acknowledges that while the most recent insulin dose change is crucial, previous doses can still exert influence on current glucose levels.

    The high confidence level (0.90) is justified given the strong biological basis for this hypothesis. However, it's important to note that while this model is more comprehensive, glucose regulation is highly complex and can be influenced by many factors not captured here, such as individual insulin sensitivity, concurrent medications, meal timing, and physical activity.

    Future refinements of this hypothesis might consider:
    1. Incorporating non-linear relationships between insulin doses and glucose responses.
    2. Accounting for individual patient characteristics that might affect insulin action.
    3. Integrating other relevant factors like carbohydrate intake or physical activity levels.

    Despite these potential areas for improvement, this hypothesis provides a strong foundation for modeling the time-dependent effects of insulin on glucose levels, which could significantly enhance the accuracy of glucose predictions in diabetes management.
    </explanation>

    <validity>
    True
    </validity>

    
    **Example 2**:
    <reasoning>
    This hypothesis suggests a relationship between the day of the week (weekend vs. weekday) and glucose measurements. 

    Positive aspects:
    1. **Observable pattern**: The rationale indicates an observed increase in mean glucose levels during weekends, providing some data-driven basis for the hypothesis.
    2. **Potential lifestyle factor**: Weekends could be associated with changes in routine that might affect glucose levels.

    Negative aspects:
    1. **Lack of clear causal mechanism**: The hypothesis doesn't provide a biological or behavioral explanation for why weekend glucose measurements would be consistently higher.
    2. **Overgeneralization**: The hypothesis assumes a consistent effect across all patients, ignoring individual variability in weekend behaviors and glucose management.
    3. **Confounding factors**: The hypothesis doesn't account for other variables that might differ between weekends and weekdays (e.g., diet, physical activity, medication adherence).
    4. **Low confidence level**: The stated confidence level of 0.60 indicates significant uncertainty.

    While it's possible that weekends could be associated with lifestyle changes that affect glucose levels, the hypothesis lacks a clear causal pathway. Factors such as dietary habits, physical activity, and medication adherence, which could vary between weekdays and weekends, are not considered in this simplified hypothesis.
    </reasoning>

    <explanation>
    The hypothesis that glucose measurements taken on weekends are consistently higher than those taken on weekdays is not sufficiently supported by the provided information and lacks a clear causal mechanism. While there may be an observed slight increase in mean glucose levels during weekends in the dataset, this alone does not establish a consistent or causal relationship.

    Glucose levels are influenced by numerous factors including diet, physical activity, stress, and medication adherence - all of which could potentially vary between weekdays and weekends. However, these factors would likely vary significantly among individuals, making a consistent weekend effect unlikely.

    The low confidence level (0.60) further indicates that this hypothesis is speculative. To be valid, the hypothesis would need to propose a more specific mechanism by which weekends affect glucose levels and account for potential confounding factors. Additionally, it would need to consider individual variability in weekend behaviors and glucose management practices.

    In its current form, this hypothesis oversimplifies the complex factors influencing glucose levels and does not provide a convincing causal link between the weekend/weekday distinction and glucose measurements.
    </explanation>

    <validity>
    False
    </validity>
    """

    human_prompt = """
    Hypothesis: {hypothesis}
    Feature Metrics: {feature_metrics}
    Rationale: {rationale}
    Relevant Features: {relevant_features}
    Expected Effect: {expected_effect}
    Confidence Level: {confidence_level}
    Target Definition: {target_definition}
    """

    def process_hypothesis(hypothesis):
        formatted_prompt = human_prompt.format(
            hypothesis=hypothesis['statement'],
            feature_metrics=hypothesis['feature_metrics'],
            rationale=hypothesis['rationale'],
            relevant_features=hypothesis['relevant_features'],
            expected_effect=hypothesis['expected_effect'],
            confidence_level=hypothesis['confidence_level'],
            target_definition=target_definition
        )
        
        response = model.invoke([("system",system_prompt),("human", formatted_prompt)]).content
        
        reasoning = re.search(r'<reasoning>(.*?)</reasoning>', response, re.DOTALL)
        explanation = re.search(r'<explanation>(.*?)</explanation>', response, re.DOTALL)
        validity = re.search(r'<validity>(.*?)</validity>', response, re.DOTALL)
        
        return {
            'statement': hypothesis['statement'],
            'reasoning': reasoning.group(1).strip() if reasoning else None,
            'explanation': explanation.group(1).strip() if explanation else None,
            'validity': validity.group(1).strip().lower() == 'true' if validity else None
        }

    with ThreadPoolExecutor() as executor:
        insights = list(executor.map(process_hypothesis, hypotheses))
    
    return insights

def generate_and_analyze_hypotheses(synthetic_data, real_data, target_definition):
    """
    Generate hypotheses, prioritize them, and generate insights using LLM.
    
    Args:
        synthetic_data (pd.DataFrame): DataFrame containing synthetic patient profiles.
        real_data (pd.DataFrame): The real dataset.
        target_definition (str): The target definition for the LLM.
    
    Returns:
        List[Dict]: List of hypotheses with LLM-generated insights.
    """
    hypotheses = generate_hypotheses(synthetic_data, real_data)
    prioritized_hypotheses = prioritize_hypotheses(hypotheses)
    
    # Add feature metrics to each hypothesis
    for hypothesis in prioritized_hypotheses:
        feature = hypothesis['relevant_features'][0]
        synthetic_values = synthetic_data[feature].dropna()
        real_values = real_data[feature].dropna()
        
        hypothesis['feature_metrics'] = {
            'synthetic_mean': synthetic_values.mean(),
            'real_mean': real_values.mean(),
            'synthetic_std': synthetic_values.std(),
            'real_std': real_values.std()
        }
    
    # Generate insights using LLM
    hypotheses_with_insights = hypothesis_reasoning_insights(prioritized_hypotheses, target_definition)
    
    # Combine original hypothesis data with LLM insights
    final_hypotheses = []
    for original, insight in zip(prioritized_hypotheses, hypotheses_with_insights):
        final_hypothesis = {
            'statement': original['statement'],
            'rationale': original['rationale'],
            'relevant_features': original['relevant_features'],
            'expected_effect': original['expected_effect'],
            'confidence_level': original['confidence_level'],
            'feature_metrics': original['feature_metrics'],
            'llm_reasoning': insight['reasoning'],
            'llm_explanation': insight['explanation'],
            'llm_validity': insight['validity']
        }
        final_hypotheses.append(final_hypothesis)
    
    return final_hypotheses


