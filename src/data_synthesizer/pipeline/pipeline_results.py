from typing import TypedDict, Optional

import numpy as np
import pandas as pd

from data_evaluator.privacy_evaluation.privacy_evaluator_anonymeter import AnonymeterResults

class GenerationResults(TypedDict):
    """Typed dictionary to store generation results."""
    synthetic_data: pd.DataFrame
    generator_model : object

class ResemblanceEvaluationResults(TypedDict):
    """Dictionary type for storing evaluation results with detailed structure."""
    categorical_univariate: dict
    numerical_univariate: dict
    categorical_multivariate: dict
    numerical_multivariate: dict
    categorical_numerical_multivariate: dict


class UtilityEvaluationResults(TypedDict):
    """Dictionary type for storing utility evaluation results."""
    train_synthetic_test_real_results: pd.DataFrame
    train_real_test_real_results: pd.DataFrame
    permutation_importance_tstr : np.ndarray
    permutation_importance_trtr : np.ndarray
    shap_importance_tstr : np.ndarray
    shap_importance_trtr : np.ndarray
    confusion_matrix_tstr : np.ndarray
    confusion_matrix_trtr : np.ndarray
    accuracy_mean_tstr : float
    accuracy_mean_trtr : float

class PrivacyEvaluationResults(TypedDict):
    """Dictionary type for storing utility evaluation results."""
    dissimilarity_synthetic_real : float
    dissimilarity_synthetic_test : float
    share : float
    attribute_synthetic_prediction : dict
    attribute_real_prediction : dict

class PrivacyAnonymeterEvaluationResults(TypedDict):
    """Dictionary type for storing utility evaluation results."""
    singling_univariate : AnonymeterResults
    singling_multivariate : AnonymeterResults
    linkability_attacks : AnonymeterResults


class PipelineResults(TypedDict) :
    generation_results: Optional[GenerationResults] = None 
    resemblance_evaluation_results: Optional[ResemblanceEvaluationResults] = None
    utility_evaluation_results: Optional[UtilityEvaluationResults] = None
    privacy_evaluation_results: Optional[PrivacyEvaluationResults] = None
    privacy_anonymeter_results: Optional[PrivacyAnonymeterEvaluationResults] = None