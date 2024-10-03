import numpy as np
import yaml
import pandas as pd
from datetime import timedelta

# Load configuration
with open('configs/morph_model.yaml', 'r') as file:
    config = yaml.safe_load(file)

def morph_profile(profile, direction, step_size=None):
    """
    Morph a given patient profile in the specified direction to change the predicted outcome.
    
    Args:
        profile (dict): The original patient profile.
        direction (str): Direction to morph ('increase' or 'decrease' predicted outcome).
        step_size (float): The step size for morphing.
    
    Returns:
        dict: The morphed patient profile.
    """
    if step_size is None:
        step_size = config['morphing_params']['step_size']
    
    morphed_profile = profile.copy()
    for feature, value in profile.items():
        if isinstance(value, (int, float)):
            if direction == 'increase':
                morphed_profile[feature] += step_size
            elif direction == 'decrease':
                morphed_profile[feature] -= step_size
        elif isinstance(value, pd.Timestamp):
            if direction == 'increase':
                morphed_profile[feature] += timedelta(hours=step_size)
            elif direction == 'decrease':
                morphed_profile[feature] -= timedelta(hours=step_size)
        elif isinstance(value, str):
            # For categorical features, we might want to implement a different strategy
            # For now, we'll leave them unchanged
            pass
        else:
            # For other data types, leave them unchanged
            pass
    
    return morphed_profile

def enforce_constraints(profile):
    """
    Ensure the morphed profile remains realistic by enforcing constraints.
    
    Args:
        profile (dict): The morphed patient profile.
    
    Returns:
        dict: The constrained patient profile.
    """
    constraints = config['morphing_params']['constraints']
    constrained_profile = profile.copy()
    
    for feature, (min_val, max_val) in constraints.items():
        if feature in constrained_profile:
            value = constrained_profile[feature]
            if isinstance(value, (int, float)):
                constrained_profile[feature] = np.clip(value, min_val, max_val)
            elif isinstance(value, pd.Timestamp):
                constrained_profile[feature] = pd.Timestamp(min(max(value, pd.Timestamp(min_val)), pd.Timestamp(max_val)))
    
    return constrained_profile

def calculate_similarity(original_profile, morphed_profile):
    """
    Calculate the similarity between the original and morphed profiles.
    
    Args:
        original_profile (dict): The original patient profile.
        morphed_profile (dict): The morphed patient profile.
    
    Returns:
        float: The similarity score.
    """
    original_values = []
    morphed_values = []
    
    for feature in original_profile:
        if isinstance(original_profile[feature], (int, float)) and not pd.isna(original_profile[feature]):
            original_values.append(original_profile[feature])
            morphed_values.append(morphed_profile[feature])
        elif isinstance(original_profile[feature], pd.Timestamp):
            original_values.append(original_profile[feature].timestamp())
            morphed_values.append(morphed_profile[feature].timestamp())
    
    original_values = np.array(original_values)
    morphed_values = np.array(morphed_values)
    
    if len(original_values) > 0:
        similarity = np.dot(original_values, morphed_values) / (np.linalg.norm(original_values) * np.linalg.norm(morphed_values))
        return float(similarity)  # Convert to float to ensure it's not a numpy type
    else:
        return 1.0  # If no valid numeric features, consider profiles identical