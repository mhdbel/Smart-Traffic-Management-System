# utils/model_utils.py
"""
Machine Learning model utilities for the Smart Traffic Management System.

This module provides functions for:
- Model training and evaluation
- Model persistence (save/load)
- Feature importance analysis
- Cross-validation
- Hyperparameter tuning
- Prediction utilities
"""

import json
import logging
import os
import pickle
import warnings
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd

# Scikit-learn imports
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error,
    explained_variance_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import (
    cross_val_score,
    GridSearchCV,
    RandomizedSearchCV,
    train_test_split,
    TimeSeriesSplit,
)
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS
# =============================================================================

DEFAULT_MODEL_DIR = Path("models")
SUPPORTED_FORMATS = ['joblib', 'pickle']


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class RegressionMetrics:
    """Metrics for regression model evaluation."""
    mse: float
    rmse: float
    mae: float
    mape: float
    r2: float
    explained_variance: float
    n_samples: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def __str__(self) -> str:
        return (
            f"Regression Metrics:\n"
            f"  MSE:  {self.mse:.4f}\n"
            f"  RMSE: {self.rmse:.4f}\n"
            f"  MAE:  {self.mae:.4f}\n"
            f"  MAPE: {self.mape:.2%}\n"
            f"  R²:   {self.r2:.4f}\n"
            f"  Explained Variance: {self.explained_variance:.4f}\n"
            f"  Samples: {self.n_samples}"
        )


@dataclass
class ClassificationMetrics:
    """Metrics for classification model evaluation."""
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: Optional[float]
    confusion_matrix: List[List[int]]
    classification_report: str
    n_samples: int
    n_classes: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def __str__(self) -> str:
        return (
            f"Classification Metrics:\n"
            f"  Accuracy:  {self.accuracy:.4f}\n"
            f"  Precision: {self.precision:.4f}\n"
            f"  Recall:    {self.recall:.4f}\n"
            f"  F1 Score:  {self.f1:.4f}\n"
            f"  ROC AUC:   {self.roc_auc:.4f if self.roc_auc else 'N/A'}\n"
            f"  Samples:   {self.n_samples}\n"
            f"  Classes:   {self.n_classes}"
        )


@dataclass
class ModelInfo:
    """Information about a saved model."""
    name: str
    model_type: str
    version: str
    created_at: str
    metrics: Dict[str, float]
    features: List[str]
    hyperparameters: Dict[str, Any]
    training_samples: int
    file_path: str
    file_size_mb: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ModelInfo':
        return cls(**data)


@dataclass 
class CrossValidationResult:
    """Results from cross-validation."""
    scores: List[float]
    mean_score: float
    std_score: float
    n_folds: int
    scoring: str
    
    def __str__(self) -> str:
        return (
            f"Cross-Validation ({self.n_folds} folds, {self.scoring}):\n"
            f"  Mean: {self.mean_score:.4f} (+/- {self.std_score:.4f})\n"
            f"  Scores: {[f'{s:.4f}' for s in self.scores]}"
        )


# =============================================================================
# EVALUATION FUNCTIONS
# =============================================================================

def evaluate_regression(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_weight: Optional[np.ndarray] = None
) -> RegressionMetrics:
    """
    Comprehensive evaluation for regression models.
    
    Args:
        y_true: True target values
        y_pred: Predicted values
        sample_weight: Optional sample weights
        
    Returns:
        RegressionMetrics with all computed metrics
        
    Raises:
        ValueError: If inputs are invalid
        
    Examples:
        >>> metrics = evaluate_regression(y_true, y_pred)
        >>> print(metrics)
        >>> print(f"RMSE: {metrics.rmse:.2f}")
    """
    # Validate inputs
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}"
        )
    
    if len(y_true) == 0:
        raise ValueError("Empty arrays provided")
    
    # Calculate metrics
    mse = mean_squared_error(y_true, y_pred, sample_weight=sample_weight)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred, sample_weight=sample_weight)
    
    # Handle MAPE (avoid division by zero)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            mape = mean_absolute_percentage_error(y_true, y_pred, sample_weight=sample_weight)
        except:
            # Manual calculation handling zeros
            mask = y_true != 0
            if mask.sum() > 0:
                mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))
            else:
                mape = float('inf')
    
    r2 = r2_score(y_true, y_pred, sample_weight=sample_weight)
    explained_var = explained_variance_score(y_true, y_pred, sample_weight=sample_weight)
    
    metrics = RegressionMetrics(
        mse=float(mse),
        rmse=float(rmse),
        mae=float(mae),
        mape=float(mape),
        r2=float(r2),
        explained_variance=float(explained_var),
        n_samples=len(y_true)
    )
    
    logger.info(f"Regression evaluation complete: RMSE={rmse:.4f}, R²={r2:.4f}")
    
    return metrics


def evaluate_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    average: str = 'weighted'
) -> ClassificationMetrics:
    """
    Comprehensive evaluation for classification models.
    
    Args:
        y_true: True class labels
        y_pred: Predicted class labels
        y_prob: Optional prediction probabilities (for ROC AUC)
        average: Averaging method ('binary', 'micro', 'macro', 'weighted')
        
    Returns:
        ClassificationMetrics with all computed metrics
        
    Examples:
        >>> metrics = evaluate_classification(y_true, y_pred, y_prob)
        >>> print(metrics)
        >>> print(f"F1: {metrics.f1:.2f}")
    """
    # Validate inputs
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}"
        )
    
    n_classes = len(np.unique(y_true))
    is_binary = n_classes == 2
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    
    # For binary classification, use binary averaging
    avg = 'binary' if is_binary else average
    
    precision = precision_score(y_true, y_pred, average=avg, zero_division=0)
    recall = recall_score(y_true, y_pred, average=avg, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=avg, zero_division=0)
    
    # ROC AUC (requires probabilities)
    roc_auc = None
    if y_prob is not None:
        try:
            if is_binary:
                # For binary, use probability of positive class
                prob = y_prob[:, 1] if y_prob.ndim == 2 else y_prob
                roc_auc = roc_auc_score(y_true, prob)
            else:
                roc_auc = roc_auc_score(y_true, y_prob, multi_class='ovr', average=average)
        except Exception as e:
            logger.warning(f"Could not compute ROC AUC: {e}")
    
    # Confusion matrix and classification report
    cm = confusion_matrix(y_true, y_pred).tolist()
    report = classification_report(y_true, y_pred, zero_division=0)
    
    metrics = ClassificationMetrics(
        accuracy=float(accuracy),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        roc_auc=float(roc_auc) if roc_auc is not None else None,
        confusion_matrix=cm,
        classification_report=report,
        n_samples=len(y_true),
        n_classes=n_classes
    )
    
    logger.info(f"Classification evaluation complete: Accuracy={accuracy:.4f}, F1={f1:.4f}")
    
    return metrics


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task: str = 'auto',
    **kwargs
) -> Union[RegressionMetrics, ClassificationMetrics]:
    """
    Unified model evaluation function.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        task: 'regression', 'classification', or 'auto'
        **kwargs: Additional arguments passed to specific evaluator
        
    Returns:
        Appropriate metrics object
        
    Examples:
        >>> # Auto-detect task type
        >>> metrics = evaluate_model(y_true, y_pred)
        
        >>> # Explicit task type
        >>> metrics = evaluate_model(y_true, y_pred, task='regression')
    """
    y_true = np.asarray(y_true).flatten()
    
    # Auto-detect task type
    if task == 'auto':
        unique_values = np.unique(y_true)
        # Heuristic: if few unique values and all integers, likely classification
        if len(unique_values) <= 20 and np.all(y_true == y_true.astype(int)):
            task = 'classification'
        else:
            task = 'regression'
        logger.info(f"Auto-detected task type: {task}")
    
    if task == 'regression':
        return evaluate_regression(y_true, y_pred, **kwargs)
    elif task == 'classification':
        return evaluate_classification(y_true, y_pred, **kwargs)
    else:
        raise ValueError(f"Unknown task: {task}. Use 'regression' or 'classification'")


# =============================================================================
# CROSS-VALIDATION
# =============================================================================

def cross_validate(
    model: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
    scoring: str = 'neg_mean_squared_error',
    time_series: bool = False,
    n_jobs: int = -1
) -> CrossValidationResult:
    """
    Perform cross-validation on a model.
    
    Args:
        model: Scikit-learn compatible model
        X: Features
        y: Target
        cv: Number of folds
        scoring: Scoring metric
        time_series: Use TimeSeriesSplit if True
        n_jobs: Number of parallel jobs (-1 for all CPUs)
        
    Returns:
        CrossValidationResult with scores
        
    Examples:
        >>> result = cross_validate(model, X, y, cv=5)
        >>> print(f"Mean score: {result.mean_score:.4f}")
    """
    if time_series:
        cv_splitter = TimeSeriesSplit(n_splits=cv)
    else:
        cv_splitter = cv
    
    logger.info(f"Running {cv}-fold cross-validation with {scoring}")
    
    scores = cross_val_score(
        model, X, y,
        cv=cv_splitter,
        scoring=scoring,
        n_jobs=n_jobs
    )
    
    # Handle negative scores (sklearn convention)
    if scoring.startswith('neg_'):
        scores = -scores
        scoring_name = scoring.replace('neg_', '')
    else:
        scoring_name = scoring
    
    result = CrossValidationResult(
        scores=scores.tolist(),
        mean_score=float(np.mean(scores)),
        std_score=float(np.std(scores)),
        n_folds=cv,
        scoring=scoring_name
    )
    
    logger.info(f"CV Result: {result.mean_score:.4f} (+/- {result.std_score:.4f})")
    
    return result


# =============================================================================
# MODEL PERSISTENCE
# =============================================================================

def save_model(
    model: BaseEstimator,
    filepath: Union[str, Path],
    metadata: Optional[Dict] = None,
    format: str = 'joblib'
) -> ModelInfo:
    """
    Save a trained model to disk.
    
    Args:
        model: Trained model to save
        filepath: Path to save the model
        metadata: Optional metadata to save alongside
        format: 'joblib' or 'pickle'
        
    Returns:
        ModelInfo about the saved model
        
    Examples:
        >>> info = save_model(model, 'models/traffic_model.pkl')
        >>> print(f"Saved to: {info.file_path}")
    """
    filepath = Path(filepath)
    
    if format not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported format: {format}. Use {SUPPORTED_FORMATS}")
    
    # Ensure directory exists
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Save model
    if format == 'joblib':
        joblib.dump(model, filepath)
    else:
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
    
    # Get file size
    file_size_mb = filepath.stat().st_size / (1024 * 1024)
    
    # Extract model info
    model_type = type(model).__name__
    
    # Try to get hyperparameters
    try:
        hyperparameters = model.get_params()
    except:
        hyperparameters = {}
    
    # Try to get feature names
    features = []
    if hasattr(model, 'feature_names_in_'):
        features = list(model.feature_names_in_)
    
    # Create model info
    info = ModelInfo(
        name=filepath.stem,
        model_type=model_type,
        version="1.0.0",
        created_at=datetime.now().isoformat(),
        metrics=metadata.get('metrics', {}) if metadata else {},
        features=features,
        hyperparameters=hyperparameters,
        training_samples=metadata.get('training_samples', 0) if metadata else 0,
        file_path=str(filepath),
        file_size_mb=round(file_size_mb, 2)
    )
    
    # Save metadata
    metadata_path = filepath.with_suffix('.json')
    with open(metadata_path, 'w') as f:
        json.dump(info.to_dict(), f, indent=2, default=str)
    
    logger.info(f"Model saved to {filepath} ({file_size_mb:.2f} MB)")
    
    return info


def load_model(
    filepath: Union[str, Path],
    format: str = 'auto'
) -> Tuple[BaseEstimator, Optional[ModelInfo]]:
    """
    Load a model from disk.
    
    Args:
        filepath: Path to the model file
        format: 'joblib', 'pickle', or 'auto'
        
    Returns:
        Tuple of (model, metadata)
        
    Raises:
        FileNotFoundError: If model file doesn't exist
        
    Examples:
        >>> model, info = load_model('models/traffic_model.pkl')
        >>> predictions = model.predict(X_test)
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Model file not found: {filepath}")
    
    # Auto-detect format
    if format == 'auto':
        format = 'joblib'  # Default to joblib
    
    # Load model
    if format == 'joblib':
        model = joblib.load(filepath)
    else:
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
    
    # Try to load metadata
    info = None
    metadata_path = filepath.with_suffix('.json')
    if metadata_path.exists():
        try:
            with open(metadata_path, 'r') as f:
                info = ModelInfo.from_dict(json.load(f))
        except Exception as e:
            logger.warning(f"Could not load model metadata: {e}")
    
    logger.info(f"Model loaded from {filepath}")
    
    return model, info


def list_models(
    directory: Union[str, Path] = None
) -> List[ModelInfo]:
    """
    List all saved models in a directory.
    
    Args:
        directory: Directory to search (default: DEFAULT_MODEL_DIR)
        
    Returns:
        List of ModelInfo for each found model
    """
    directory = Path(directory) if directory else DEFAULT_MODEL_DIR
    
    if not directory.exists():
        return []
    
    models = []
    
    for metadata_file in directory.glob('*.json'):
        try:
            with open(metadata_file, 'r') as f:
                info = ModelInfo.from_dict(json.load(f))
                models.append(info)
        except Exception as e:
            logger.warning(f"Could not read {metadata_file}: {e}")
    
    # Sort by creation date (newest first)
    models.sort(key=lambda x: x.created_at, reverse=True)
    
    return models


def delete_model(
    filepath: Union[str, Path]
) -> bool:
    """
    Delete a saved model and its metadata.
    
    Args:
        filepath: Path to the model file
        
    Returns:
        True if deleted successfully
    """
    filepath = Path(filepath)
    
    deleted = False
    
    if filepath.exists():
        filepath.unlink()
        deleted = True
        logger.info(f"Deleted model: {filepath}")
    
    # Also delete metadata
    metadata_path = filepath.with_suffix('.json')
    if metadata_path.exists():
        metadata_path.unlink()
        logger.info(f"Deleted metadata: {metadata_path}")
    
    return deleted


# =============================================================================
# FEATURE IMPORTANCE
# =============================================================================

def get_feature_importance(
    model: BaseEstimator,
    feature_names: Optional[List[str]] = None,
    top_n: Optional[int] = None
) -> pd.DataFrame:
    """
    Extract feature importance from a trained model.
    
    Args:
        model: Trained model with feature_importances_ or coef_
        feature_names: Optional feature names
        top_n: Return only top N features
        
    Returns:
        DataFrame with feature importances
        
    Examples:
        >>> importance = get_feature_importance(model, feature_names)
        >>> print(importance.head(10))
    """
    # Try different attributes
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_).flatten()
    elif hasattr(model, 'named_steps'):
        # Pipeline - get from last step
        last_step = list(model.named_steps.values())[-1]
        return get_feature_importance(last_step, feature_names, top_n)
    else:
        raise ValueError("Model doesn't have feature_importances_ or coef_")
    
    # Get feature names
    if feature_names is None:
        if hasattr(model, 'feature_names_in_'):
            feature_names = list(model.feature_names_in_)
        else:
            feature_names = [f'feature_{i}' for i in range(len(importances))]
    
    # Create DataFrame
    df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    })
    
    # Sort by importance
    df = df.sort_values('importance', ascending=False).reset_index(drop=True)
    
    # Normalize
    df['importance_pct'] = df['importance'] / df['importance'].sum() * 100
    df['cumulative_pct'] = df['importance_pct'].cumsum()
    
    if top_n:
        df = df.head(top_n)
    
    return df


def plot_feature_importance(
    model: BaseEstimator,
    feature_names: Optional[List[str]] = None,
    top_n: int = 20,
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None
) -> None:
    """
    Plot feature importance as a horizontal bar chart.
    
    Args:
        model: Trained model
        feature_names: Feature names
        top_n: Number of top features to show
        figsize: Figure size
        save_path: Optional path to save the plot
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.error("matplotlib is required for plotting")
        return
    
    importance_df = get_feature_importance(model, feature_names, top_n)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot horizontal bars
    y_pos = range(len(importance_df))
    ax.barh(y_pos, importance_df['importance_pct'], align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(importance_df['feature'])
    ax.invert_yaxis()
    ax.set_xlabel('Importance (%)')
    ax.set_title(f'Top {top_n} Feature Importances')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved feature importance plot to {save_path}")
    
    plt.show()


# =============================================================================
# HYPERPARAMETER TUNING
# =============================================================================

def tune_hyperparameters(
    model: BaseEstimator,
    param_grid: Dict[str, List],
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
    scoring: str = 'neg_mean_squared_error',
    method: str = 'grid',
    n_iter: int = 100,
    n_jobs: int = -1,
    verbose: int = 1
) -> Tuple[BaseEstimator, Dict[str, Any], float]:
    """
    Tune model hyperparameters using grid or random search.
    
    Args:
        model: Model to tune
        param_grid: Parameter grid or distributions
        X: Features
        y: Target
        cv: Number of CV folds
        scoring: Scoring metric
        method: 'grid' or 'random'
        n_iter: Number of iterations for random search
        n_jobs: Number of parallel jobs
        verbose: Verbosity level
        
    Returns:
        Tuple of (best_model, best_params, best_score)
        
    Examples:
        >>> param_grid = {
        ...     'n_estimators': [100, 200, 300],
        ...     'max_depth': [3, 5, 7, 10]
        ... }
        >>> best_model, best_params, best_score = tune_hyperparameters(
        ...     model, param_grid, X_train, y_train
        ... )
    """
    logger.info(f"Starting hyperparameter tuning with {method} search")
    
    if method == 'grid':
        search = GridSearchCV(
            model,
            param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=verbose,
            return_train_score=True
        )
    elif method == 'random':
        search = RandomizedSearchCV(
            model,
            param_grid,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=verbose,
            random_state=42,
            return_train_score=True
        )
    else:
        raise ValueError(f"Unknown method: {method}. Use 'grid' or 'random'")
    
    search.fit(X, y)
    
    best_model = search.best_estimator_
    best_params = search.best_params_
    best_score = -search.best_score_ if scoring.startswith('neg_') else search.best_score_
    
    logger.info(f"Best score: {best_score:.4f}")
    logger.info(f"Best params: {best_params}")
    
    return best_model, best_params, best_score


# =============================================================================
# PREDICTION UTILITIES
# =============================================================================

def predict_with_confidence(
    model: BaseEstimator,
    X: np.ndarray,
    confidence: float = 0.95
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Make predictions with confidence intervals (for ensemble models).
    
    Args:
        model: Trained model (works best with ensemble models)
        X: Features to predict
        confidence: Confidence level (0.95 = 95%)
        
    Returns:
        Tuple of (predictions, lower_bound, upper_bound)
        
    Note:
        For non-ensemble models, bounds equal predictions.
    """
    from scipy import stats
    
    predictions = model.predict(X)
    
    # Check if model has individual estimators (ensemble)
    if hasattr(model, 'estimators_'):
        # Get predictions from all estimators
        all_preds = np.array([
            estimator.predict(X) for estimator in model.estimators_
        ])
        
        # Calculate confidence intervals
        mean_pred = np.mean(all_preds, axis=0)
        std_pred = np.std(all_preds, axis=0)
        
        z_score = stats.norm.ppf((1 + confidence) / 2)
        lower = mean_pred - z_score * std_pred
        upper = mean_pred + z_score * std_pred
        
        return predictions, lower, upper
    else:
        # No confidence intervals available
        return predictions, predictions, predictions


def batch_predict(
    model: BaseEstimator,
    X: np.ndarray,
    batch_size: int = 1000,
    verbose: bool = True
) -> np.ndarray:
    """
    Make predictions in batches (for large datasets).
    
    Args:
        model: Trained model
        X: Features (can be large)
        batch_size: Size of each batch
        verbose: Show progress
        
    Returns:
        Predictions array
    """
    n_samples = len(X)
    predictions = []
    
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    for i in range(n_batches):
        start = i * batch_size
        end = min(start + batch_size, n_samples)
        
        batch_pred = model.predict(X[start:end])
        predictions.append(batch_pred)
        
        if verbose and (i + 1) % 10 == 0:
            logger.info(f"Processed {end}/{n_samples} samples")
    
    return np.concatenate(predictions)


# =============================================================================
# MODEL COMPARISON
# =============================================================================

def compare_models(
    models: Dict[str, BaseEstimator],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    task: str = 'regression'
) -> pd.DataFrame:
    """
    Compare multiple models on the same dataset.
    
    Args:
        models: Dictionary of {name: model}
        X_train, y_train: Training data
        X_test, y_test: Test data
        task: 'regression' or 'classification'
        
    Returns:
        DataFrame comparing model performance
        
    Examples:
        >>> models = {
        ...     'Random Forest': RandomForestRegressor(),
        ...     'Gradient Boosting': GradientBoostingRegressor(),
        ...     'Linear Regression': LinearRegression()
        ... }
        >>> results = compare_models(models, X_train, y_train, X_test, y_test)
    """
    results = []
    
    for name, model in models.items():
        logger.info(f"Training {name}...")
        
        # Train
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Evaluate
        metrics = evaluate_model(y_test, y_pred, task=task)
        
        # Collect results
        result = {'model': name}
        result.update(metrics.to_dict())
        results.append(result)
    
    df = pd.DataFrame(results)
    
    # Sort by primary metric
    if task == 'regression':
        df = df.sort_values('rmse')
    else:
        df = df.sort_values('f1', ascending=False)
    
    return df.reset_index(drop=True)


# =============================================================================
# PIPELINE UTILITIES
# =============================================================================

def create_pipeline(
    model: BaseEstimator,
    scaler: str = 'standard',
    feature_selector: Optional[BaseEstimator] = None
) -> Pipeline:
    """
    Create a preprocessing + model pipeline.
    
    Args:
        model: Model to use
        scaler: 'standard', 'minmax', or None
        feature_selector: Optional feature selector
        
    Returns:
        Scikit-learn Pipeline
        
    Examples:
        >>> pipeline = create_pipeline(
        ...     RandomForestRegressor(),
        ...     scaler='standard'
        ... )
        >>> pipeline.fit(X_train, y_train)
    """
    steps = []
    
    # Add scaler
    if scaler == 'standard':
        steps.append(('scaler', StandardScaler()))
    elif scaler == 'minmax':
        steps.append(('scaler', MinMaxScaler()))
    
    # Add feature selector
    if feature_selector is not None:
        steps.append(('selector', feature_selector))
    
    # Add model
    steps.append(('model', model))
    
    return Pipeline(steps)


# =============================================================================
# BACKWARD COMPATIBILITY
# =============================================================================

# Keep original function signature for backward compatibility
def evaluate_model_legacy(y_true, y_pred):
    """
    Legacy evaluation function for backward compatibility.
    
    Deprecated: Use evaluate_regression() or evaluate_model() instead.
    """
    import warnings
    warnings.warn(
        "evaluate_model_legacy is deprecated. Use evaluate_model() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    metrics = evaluate_regression(y_true, y_pred)
    print(metrics)
    return metrics


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Data classes
    'RegressionMetrics',
    'ClassificationMetrics',
    'ModelInfo',
    'CrossValidationResult',
    # Evaluation
    'evaluate_regression',
    'evaluate_classification',
    'evaluate_model',
    # Cross-validation
    'cross_validate',
    # Model persistence
    'save_model',
    'load_model',
    'list_models',
    'delete_model',
    # Feature importance
    'get_feature_importance',
    'plot_feature_importance',
    # Hyperparameter tuning
    'tune_hyperparameters',
    # Prediction
    'predict_with_confidence',
    'batch_predict',
    # Comparison
    'compare_models',
    # Pipeline
    'create_pipeline',
]
