from pydantic import BaseModel, Field
from typing import List, Dict, Optional

class PatientProfile(BaseModel):
    """
    Represents a patient profile with various features.
    """
    features: Dict[str, float] = Field(..., description="Dictionary of feature names and their values")

class Hypothesis(BaseModel):
    """
    Represents a generated hypothesis.
    """
    statement: str = Field(..., description="Hypothesis statement")
    rationale: str = Field(..., description="Rationale for the hypothesis")
    relevant_features: List[str] = Field(..., description="List of relevant features")
    expected_effect: str = Field(..., description="Expected effect (increase, decrease, difference)")
    confidence_level: float = Field(..., description="Confidence level of the hypothesis")

class ModelEvaluation(BaseModel):
    """
    Represents the evaluation metrics of a predictive model.
    """
    accuracy: Optional[float] = Field(None, description="Accuracy of the model")
    precision: Optional[float] = Field(None, description="Precision of the model")
    recall: Optional[float] = Field(None, description="Recall of the model")
    f1_score: Optional[float] = Field(None, description="F1 score of the model")
    roc_auc: Optional[float] = Field(None, description="ROC AUC score of the model")

class FeatureImportance(BaseModel):
    """
    Represents the feature importance scores.
    """
    feature_scores: Dict[str, float] = Field(..., description="Dictionary of feature names and their importance scores")

class SyntheticProfileGeneration(BaseModel):
    """
    Represents the process of generating synthetic patient profiles.
    """
    vae_model: str = Field(..., description="Path to the trained VAE model")
    num_profiles: int = Field(..., description="Number of synthetic profiles generated")
    synthetic_profiles: List[PatientProfile] = Field(..., description="List of generated synthetic patient profiles")

class HypothesisValidationResult(BaseModel):
    """
    Represents the result of hypothesis validation.
    """
    hypothesis: Hypothesis = Field(..., description="The hypothesis being validated")
    validation_result: bool = Field(..., description="Result of the validation (True if validated, False otherwise)")
    supporting_data: Dict[str, float] = Field(..., description="Supporting data for the validation result")