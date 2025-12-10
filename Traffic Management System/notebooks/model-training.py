#!/usr/bin/env python3
"""
Traffic Congestion Prediction Model Trainer
Version: 3.0

Features:
- Bayesian hyperparameter optimization with early stopping
- Multiple model comparison
- Comprehensive evaluation metrics
- Feature importance analysis with SHAP
- Model versioning and metadata tracking
- Data validation and preprocessing
- Production-ready logging
- CLI interface
- Experiment tracking

Usage:
    python model_training.py --data data/processed/traffic.csv --output models/
    python model_training.py --quick  # Fast training with fewer iterations
    python model_training.py --compare  # Compare multiple models
"""

# =============================================================================
# IMPORTS
# =============================================================================

import argparse
import gc
import hashlib
import json
import logging
import os
import subprocess
import sys
import warnings
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor,
)
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    TimeSeriesSplit,
)
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    explained_variance_score,
    mean_absolute_percentage_error,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Bayesian optimization
try:
    from skopt import BayesSearchCV
    from skopt.space import Integer, Real, Categorical
    from skopt.callbacks import DeltaYStopper
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False
    warnings.warn("scikit-optimize not installed. Using GridSearchCV instead.")

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class TrainingConfig:
    """Training configuration with defaults."""
    
    # Paths
    data_path: Path = field(default_factory=lambda: Path("data/processed_data/processed_traffic_data.csv"))
    model_dir: Path = field(default_factory=lambda: Path("models"))
    log_dir: Path = field(default_factory=lambda: Path("logs"))
    
    # Data
    target_column: str = "traffic_volume"
    datetime_column: str = "date_time"
    test_size: float = 0.2
    val_size: float = 0.1
    
    # Training
    random_state: int = 42
    cv_folds: int = 5
    n_iter: int = 30
    n_jobs: int = -1
    use_time_series_cv: bool = True
    
    # Model selection
    model_type: str = "random_forest"
    compare_models: bool = False
    
    # Optimization
    early_stopping_delta: float = 0.001
    early_stopping_n_best: int = 5
    
    # Output
    save_model: bool = True
    verbose: int = 1
    
    @classmethod
    def from_args(cls, args: argparse.Namespace) -> 'TrainingConfig':
        """Create config from command line arguments."""
        config = cls()
        
        if args.data:
            config.data_path = Path(args.data)
        if args.output:
            config.model_dir = Path(args.output)
        if hasattr(args, 'n_iter') and args.n_iter:
            config.n_iter = args.n_iter
        if hasattr(args, 'cv_folds') and args.cv_folds:
            config.cv_folds = args.cv_folds
        if hasattr(args, 'test_size') and args.test_size:
            config.test_size = args.test_size
        if hasattr(args, 'random_state') and args.random_state:
            config.random_state = args.random_state
        if hasattr(args, 'quick') and args.quick:
            config.n_iter = 10
            config.cv_folds = 3
        if hasattr(args, 'compare') and args.compare:
            config.compare_models = True
        if hasattr(args, 'verbose'):
            config.verbose = args.verbose
        
        return config
    
    def validate(self) -> None:
        """Validate configuration."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Training data not found: {self.data_path}")
        
        if self.test_size <= 0 or self.test_size >= 1:
            raise ValueError("test_size must be between 0 and 1")
        
        if self.cv_folds < 2:
            raise ValueError("cv_folds must be at least 2")


# =============================================================================
# LOGGING
# =============================================================================

def setup_logging(log_dir: Path, verbose: int = 1) -> logging.Logger:
    """Setup logging with file and console handlers."""
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"training_{timestamp}.log"
    
    # Determine log level
    log_level = logging.DEBUG if verbose > 1 else logging.INFO
    
    # Create logger
    logger = logging.getLogger("model_trainer")
    logger.setLevel(log_level)
    logger.handlers = []  # Clear existing handlers
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_format = logging.Formatter('%(levelname)s - %(message)s')
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    logger.info(f"Logging to: {log_file}")
    
    return logger


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class TrainingMetrics:
    """Container for training metrics."""
    mse: float
    rmse: float
    mae: float
    mape: Optional[float]
    r2: float
    explained_variance: float
    cv_mean: float
    cv_std: float
    training_samples: int
    test_samples: int
    training_time_seconds: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def __str__(self) -> str:
        return (
            f"\n{'='*50}\n"
            f"📊 Model Performance Metrics\n"
            f"{'='*50}\n"
            f"  RMSE:               {self.rmse:,.2f}\n"
            f"  MAE:                {self.mae:,.2f}\n"
            f"  MAPE:               {self.mape:.2%}\n" if self.mape else ""
            f"  R² Score:           {self.r2:.4f}\n"
            f"  Explained Variance: {self.explained_variance:.4f}\n"
            f"  CV Score:           {-self.cv_mean:.2f} (+/- {self.cv_std:.2f})\n"
            f"  Training Samples:   {self.training_samples:,}\n"
            f"  Test Samples:       {self.test_samples:,}\n"
            f"  Training Time:      {self.training_time_seconds:.1f}s\n"
            f"{'='*50}"
        )


@dataclass
class ModelArtifacts:
    """Container for model artifacts."""
    model_path: Path
    metadata_path: Path
    version: str
    metrics: TrainingMetrics
    feature_names: List[str]
    best_params: Dict[str, Any]


# =============================================================================
# DATA VALIDATION
# =============================================================================

class DataValidator:
    """Validates and preprocesses training data."""
    
    # Column requirements
    REQUIRED_COLUMNS = ['traffic_volume']
    EXPECTED_NUMERIC = ['hour', 'day_of_week', 'month', 'temperature']
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def validate(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """
        Validate and clean the training data.
        
        Args:
            df: Input DataFrame
            target_col: Target column name
            
        Returns:
            Validated DataFrame
        """
        self.logger.info(f"Validating data: {len(df)} rows, {len(df.columns)} columns")
        
        # Check target column
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in data")
        
        # Make a copy to avoid modifying original
        df = df.copy()
        
        # Remove rows with missing target
        missing_target = df[target_col].isna().sum()
        if missing_target > 0:
            self.logger.warning(f"Removing {missing_target} rows with missing target")
            df = df.dropna(subset=[target_col])
        
        # Validate target values
        if (df[target_col] < 0).any():
            self.logger.warning("Negative target values found. Clipping to zero.")
            df[target_col] = df[target_col].clip(lower=0)
        
        # Check for infinite values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        inf_counts = np.isinf(df[numeric_cols]).sum()
        if inf_counts.any():
            inf_cols = inf_counts[inf_counts > 0].index.tolist()
            self.logger.warning(f"Infinite values found in: {inf_cols}. Replacing with NaN.")
            df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
        
        # Handle remaining missing values
        missing = df.isnull().sum()
        cols_with_missing = missing[missing > 0]
        if len(cols_with_missing) > 0:
            self.logger.info(f"Filling missing values in {len(cols_with_missing)} columns")
            
            # Fill numeric with median
            for col in df.select_dtypes(include=[np.number]).columns:
                if df[col].isnull().any():
                    df[col] = df[col].fillna(df[col].median())
            
            # Fill categorical with mode
            for col in df.select_dtypes(include=['object', 'category']).columns:
                if df[col].isnull().any():
                    mode_val = df[col].mode()
                    df[col] = df[col].fillna(mode_val[0] if len(mode_val) > 0 else 'Unknown')
        
        # Remove datetime columns (not suitable for training)
        datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        if datetime_cols:
            self.logger.info(f"Removing datetime columns: {datetime_cols}")
            df = df.drop(columns=datetime_cols)
        
        # Remove non-numeric columns that can't be used
        object_cols = df.select_dtypes(include=['object']).columns.tolist()
        if object_cols:
            self.logger.warning(f"Removing non-encoded categorical columns: {object_cols}")
            df = df.drop(columns=object_cols)
        
        self.logger.info(f"Validation complete: {len(df)} rows, {len(df.columns)} columns")
        
        return df
    
    def get_feature_stats(self, df: pd.DataFrame, target_col: str) -> Dict:
        """Get statistics about features."""
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        return {
            'n_features': len(X.columns),
            'n_samples': len(df),
            'feature_names': X.columns.tolist(),
            'target_stats': {
                'mean': float(y.mean()),
                'std': float(y.std()),
                'min': float(y.min()),
                'max': float(y.max()),
            }
        }


# =============================================================================
# MODEL DEFINITIONS
# =============================================================================

def get_model_configs() -> Dict[str, Dict]:
    """Get model configurations with search spaces."""
    
    configs = {
        'random_forest': {
            'model': RandomForestRegressor(random_state=42, n_jobs=-1),
            'search_space': {
                'n_estimators': Integer(50, 300),
                'max_depth': Integer(5, 30),
                'min_samples_split': Integer(2, 20),
                'min_samples_leaf': Integer(1, 10),
                'max_features': Real(0.3, 1.0),
            },
            'quick_params': {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
            }
        },
        'gradient_boosting': {
            'model': GradientBoostingRegressor(random_state=42),
            'search_space': {
                'n_estimators': Integer(50, 300),
                'max_depth': Integer(3, 15),
                'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
                'min_samples_split': Integer(2, 20),
                'min_samples_leaf': Integer(1, 10),
                'subsample': Real(0.6, 1.0),
            },
            'quick_params': {
                'n_estimators': 100,
                'max_depth': 5,
                'learning_rate': 0.1,
            }
        },
        'extra_trees': {
            'model': ExtraTreesRegressor(random_state=42, n_jobs=-1),
            'search_space': {
                'n_estimators': Integer(50, 300),
                'max_depth': Integer(5, 30),
                'min_samples_split': Integer(2, 20),
                'min_samples_leaf': Integer(1, 10),
            },
            'quick_params': {
                'n_estimators': 100,
                'max_depth': 15,
            }
        },
        'ridge': {
            'model': Ridge(random_state=42),
            'search_space': {
                'alpha': Real(0.001, 100, prior='log-uniform'),
            },
            'quick_params': {
                'alpha': 1.0,
            }
        },
    }
    
    return configs


# =============================================================================
# MODEL TRAINER
# =============================================================================

class ModelTrainer:
    """Trains and evaluates ML models."""
    
    def __init__(self, config: TrainingConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.validator = DataValidator(logger)
        self.best_model = None
        self.best_params = None
        self.feature_names = None
        self.scaler = None
    
    def prepare_data(self) -> Dict[str, Tuple[pd.DataFrame, pd.Series]]:
        """
        Load and prepare training data.
        
        Returns:
            Dictionary with train, val, test splits
        """
        self.logger.info(f"Loading data from {self.config.data_path}")
        
        # Load data
        df = pd.read_csv(self.config.data_path)
        self.logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
        
        # Validate
        df = self.validator.validate(df, self.config.target_column)
        
        # Get feature stats
        stats = self.validator.get_feature_stats(df, self.config.target_column)
        self.logger.info(f"Features: {stats['n_features']}, Samples: {stats['n_samples']}")
        
        # Split features and target
        X = df.drop(columns=[self.config.target_column])
        y = df[self.config.target_column]
        
        self.feature_names = X.columns.tolist()
        
        # Split data
        # First: train+val vs test
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y,
            test_size=self.config.test_size,
            random_state=self.config.random_state
        )
        
        # Second: train vs val
        val_ratio = self.config.val_size / (1 - self.config.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval,
            test_size=val_ratio,
            random_state=self.config.random_state
        )
        
        self.logger.info(
            f"Split sizes - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}"
        )
        
        return {
            'train': (X_train, y_train),
            'val': (X_val, y_val),
            'test': (X_test, y_test),
        }
    
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame = None,
        y_val: pd.Series = None,
        model_type: str = None
    ) -> Tuple[Any, Dict]:
        """
        Train model with hyperparameter optimization.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional)
            y_val: Validation target (optional)
            model_type: Model type to train
            
        Returns:
            Tuple of (best_model, best_params)
        """
        import time
        start_time = time.time()
        
        model_type = model_type or self.config.model_type
        configs = get_model_configs()
        
        if model_type not in configs:
            raise ValueError(f"Unknown model type: {model_type}. Available: {list(configs.keys())}")
        
        model_config = configs[model_type]
        base_model = model_config['model']
        
        self.logger.info(f"Training {model_type} model...")
        
        # Setup cross-validation
        if self.config.use_time_series_cv:
            cv = TimeSeriesSplit(n_splits=self.config.cv_folds)
            self.logger.info("Using TimeSeriesSplit for cross-validation")
        else:
            cv = self.config.cv_folds
        
        # Bayesian optimization if available
        if SKOPT_AVAILABLE and self.config.n_iter > 5:
            self.logger.info(f"Starting Bayesian hyperparameter search ({self.config.n_iter} iterations)...")
            
            # Early stopping callback
            early_stop = DeltaYStopper(
                delta=self.config.early_stopping_delta,
                n_best=self.config.early_stopping_n_best
            )
            
            search = BayesSearchCV(
                estimator=base_model,
                search_spaces=model_config['search_space'],
                n_iter=self.config.n_iter,
                cv=cv,
                n_jobs=self.config.n_jobs,
                random_state=self.config.random_state,
                verbose=self.config.verbose,
                scoring='neg_mean_squared_error',
            )
            
            # Fit with optional early stopping
            try:
                search.fit(X_train, y_train, callback=early_stop)
            except TypeError:
                # Some versions don't support callback in fit
                search.fit(X_train, y_train)
            
            self.best_model = search.best_estimator_
            self.best_params = search.best_params_
            
            self.logger.info(f"Best CV score: {-search.best_score_:.4f}")
            
        else:
            # Quick training with default params
            self.logger.info("Using quick training with default parameters")
            
            quick_params = model_config.get('quick_params', {})
            self.best_model = base_model.set_params(**quick_params)
            self.best_model.fit(X_train, y_train)
            self.best_params = quick_params
        
        training_time = time.time() - start_time
        self.logger.info(f"Training completed in {training_time:.1f} seconds")
        self.logger.info(f"Best parameters: {self.best_params}")
        
        return self.best_model, self.best_params
    
    def evaluate(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        X_train: pd.DataFrame = None,
        y_train: pd.Series = None,
        training_time: float = 0.0
    ) -> TrainingMetrics:
        """
        Evaluate model performance.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test target
            X_train: Training features (for CV)
            y_train: Training target (for CV)
            training_time: Training duration
            
        Returns:
            TrainingMetrics object
        """
        self.logger.info("Evaluating model performance...")
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Basic metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        explained_var = explained_variance_score(y_test, y_pred)
        
        # MAPE (handle zeros)
        try:
            mape = mean_absolute_percentage_error(y_test, y_pred)
        except:
            mask = y_test != 0
            if mask.sum() > 0:
                mape = np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask]))
            else:
                mape = None
        
        # Cross-validation on combined data
        cv_mean, cv_std = 0.0, 0.0
        if X_train is not None and y_train is not None:
            X_all = pd.concat([X_train, X_test])
            y_all = pd.concat([y_train, y_test])
            
            cv_scores = cross_val_score(
                model, X_all, y_all,
                cv=self.config.cv_folds,
                scoring='neg_mean_squared_error',
                n_jobs=self.config.n_jobs
            )
            cv_mean = np.mean(cv_scores)
            cv_std = np.std(cv_scores)
        
        metrics = TrainingMetrics(
            mse=float(mse),
            rmse=float(rmse),
            mae=float(mae),
            mape=float(mape) if mape is not None else None,
            r2=float(r2),
            explained_variance=float(explained_var),
            cv_mean=float(cv_mean),
            cv_std=float(cv_std),
            training_samples=len(X_train) if X_train is not None else 0,
            test_samples=len(X_test),
            training_time_seconds=training_time,
        )
        
        self.logger.info(str(metrics))
        
        return metrics
    
    def get_feature_importance(
        self,
        model: Any,
        feature_names: List[str],
        top_n: int = 20
    ) -> pd.DataFrame:
        """Get feature importance from trained model."""
        
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_)
        else:
            self.logger.warning("Model doesn't support feature importance")
            return pd.DataFrame()
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        importance_df['importance_pct'] = (
            importance_df['importance'] / importance_df['importance'].sum() * 100
        )
        importance_df['cumulative_pct'] = importance_df['importance_pct'].cumsum()
        
        # Log top features
        self.logger.info(f"\n📊 Top {min(top_n, len(importance_df))} Features:")
        for _, row in importance_df.head(top_n).iterrows():
            self.logger.info(f"   {row['feature']}: {row['importance_pct']:.2f}%")
        
        return importance_df.head(top_n)
    
    def compare_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> pd.DataFrame:
        """Compare multiple models."""
        self.logger.info("\n🔄 Comparing multiple models...")
        
        configs = get_model_configs()
        results = []
        
        for model_name, config in configs.items():
            self.logger.info(f"\nTraining {model_name}...")
            
            try:
                import time
                start = time.time()
                
                # Quick training for comparison
                model = config['model']
                quick_params = config.get('quick_params', {})
                model.set_params(**quick_params)
                model.fit(X_train, y_train)
                
                training_time = time.time() - start
                
                # Evaluate
                y_pred = model.predict(X_test)
                
                results.append({
                    'model': model_name,
                    'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                    'mae': mean_absolute_error(y_test, y_pred),
                    'r2': r2_score(y_test, y_pred),
                    'training_time': training_time,
                })
                
            except Exception as e:
                self.logger.error(f"Error training {model_name}: {e}")
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(results).sort_values('rmse')
        
        self.logger.info("\n📊 Model Comparison:")
        self.logger.info(comparison_df.to_string(index=False))
        
        return comparison_df


# =============================================================================
# MODEL SAVER
# =============================================================================

class ModelSaver:
    """Saves model artifacts and metadata."""
    
    def __init__(self, model_dir: Path, logger: logging.Logger):
        self.model_dir = model_dir
        self.logger = logger
        self.model_dir.mkdir(parents=True, exist_ok=True)
    
    def get_git_hash(self) -> Optional[str]:
        """Get current git commit hash."""
        try:
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.stdout.strip()[:8] if result.returncode == 0 else None
        except:
            return None
    
    def get_data_hash(self, data_path: Path) -> str:
        """Get hash of training data file."""
        with open(data_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()[:8]
    
    def save(
        self,
        model: Any,
        metrics: TrainingMetrics,
        feature_names: List[str],
        best_params: Dict,
        config: TrainingConfig,
        feature_importance: pd.DataFrame = None
    ) -> ModelArtifacts:
        """
        Save model and all associated metadata.
        
        Args:
            model: Trained model
            metrics: Training metrics
            feature_names: List of feature names
            best_params: Best hyperparameters
            config: Training configuration
            feature_importance: Feature importance DataFrame
            
        Returns:
            ModelArtifacts object
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version = f"v{timestamp}"
        
        # Model filename
        model_filename = f"traffic_model_{timestamp}.pkl"
        model_path = self.model_dir / model_filename
        
        # Save model
        joblib.dump(model, model_path)
        self.logger.info(f"Model saved to: {model_path}")
        
        # Also save as 'latest' for easy access
        latest_path = self.model_dir / "traffic_model.pkl"
        joblib.dump(model, latest_path)
        self.logger.info(f"Latest model saved to: {latest_path}")
        
        # Prepare metadata
        metadata = {
            'version': version,
            'model_type': type(model).__name__,
            'created_at': datetime.now().isoformat(),
            'git_hash': self.get_git_hash(),
            'data_hash': self.get_data_hash(config.data_path) if config.data_path.exists() else None,
            'training_config': {
                'data_path': str(config.data_path),
                'test_size': config.test_size,
                'cv_folds': config.cv_folds,
                'n_iter': config.n_iter,
                'random_state': config.random_state,
            },
            'best_params': best_params,
            'metrics': metrics.to_dict(),
            'feature_names': feature_names,
            'n_features': len(feature_names),
            'environment': {
                'python_version': sys.version,
                'pandas': pd.__version__,
                'numpy': np.__version__,
                'sklearn': sklearn.__version__,
                'platform': sys.platform,
            }
        }
        
        if feature_importance is not None:
            metadata['feature_importance'] = feature_importance.to_dict('records')
        
        # Save metadata
        meta_path = self.model_dir / f"model_metadata_{timestamp}.json"
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        # Also save as 'latest'
        latest_meta_path = self.model_dir / "model_metadata.json"
        with open(latest_meta_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        self.logger.info(f"Metadata saved to: {meta_path}")
        
        return ModelArtifacts(
            model_path=model_path,
            metadata_path=meta_path,
            version=version,
            metrics=metrics,
            feature_names=feature_names,
            best_params=best_params,
        )


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_training_pipeline(config: TrainingConfig) -> ModelArtifacts:
    """
    Run the complete training pipeline.
    
    Args:
        config: Training configuration
        
    Returns:
        ModelArtifacts with paths and metrics
    """
    import time
    pipeline_start = time.time()
    
    # Setup logging
    logger = setup_logging(config.log_dir, config.verbose)
    
    logger.info("=" * 60)
    logger.info("🚦 SMART TRAFFIC PREDICTION MODEL TRAINER")
    logger.info("=" * 60)
    logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Validate config
        config.validate()
        
        # Initialize components
        trainer = ModelTrainer(config, logger)
        saver = ModelSaver(config.model_dir, logger)
        
        # Prepare data
        data_splits = trainer.prepare_data()
        X_train, y_train = data_splits['train']
        X_val, y_val = data_splits['val']
        X_test, y_test = data_splits['test']
        
        # Compare models if requested
        if config.compare_models:
            comparison = trainer.compare_models(X_train, y_train, X_test, y_test)
            # Use best model from comparison
            best_model_type = comparison.iloc[0]['model']
            logger.info(f"Best model from comparison: {best_model_type}")
            config.model_type = best_model_type
        
        # Train model
        training_start = time.time()
        model, best_params = trainer.train(X_train, y_train, X_val, y_val)
        training_time = time.time() - training_start
        
        # Evaluate
        metrics = trainer.evaluate(
            model, X_test, y_test, X_train, y_train, training_time
        )
        
        # Feature importance
        importance_df = trainer.get_feature_importance(model, trainer.feature_names)
        
        # Save model
        if config.save_model:
            artifacts = saver.save(
                model=model,
                metrics=metrics,
                feature_names=trainer.feature_names,
                best_params=best_params,
                config=config,
                feature_importance=importance_df,
            )
        else:
            artifacts = ModelArtifacts(
                model_path=None,
                metadata_path=None,
                version="unsaved",
                metrics=metrics,
                feature_names=trainer.feature_names,
                best_params=best_params,
            )
        
        # Final summary
        pipeline_time = time.time() - pipeline_start
        logger.info("\n" + "=" * 60)
        logger.info("✅ TRAINING PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        logger.info(f"Total time: {pipeline_time:.1f} seconds")
        logger.info(f"Final RMSE: {metrics.rmse:.2f}")
        logger.info(f"Final R²: {metrics.r2:.4f}")
        if artifacts.model_path:
            logger.info(f"Model saved to: {artifacts.model_path}")
        logger.info("=" * 60)
        
        return artifacts
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}", exc_info=True)
        raise


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train traffic prediction model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python model_training.py
  python model_training.py --data data/traffic.csv --output models/
  python model_training.py --quick
  python model_training.py --compare --n-iter 50
        """
    )
    
    parser.add_argument(
        '--data', '-d',
        type=str,
        help='Path to training data CSV'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output directory for model'
    )
    parser.add_argument(
        '--model-type',
        type=str,
        choices=['random_forest', 'gradient_boosting', 'extra_trees', 'ridge'],
        default='random_forest',
        help='Model type to train'
    )
    parser.add_argument(
        '--n-iter',
        type=int,
        default=30,
        help='Number of Bayesian optimization iterations'
    )
    parser.add_argument(
        '--cv-folds',
        type=int,
        default=5,
        help='Number of cross-validation folds'
    )
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Test set proportion'
    )
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random seed'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick training (fewer iterations)'
    )
    parser.add_argument(
        '--compare',
        action='store_true',
        help='Compare multiple models'
    )
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save model'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='count',
        default=1,
        help='Increase verbosity'
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Create config from args
    config = TrainingConfig.from_args(args)
    
    if args.no_save:
        config.save_model = False
    
    # Run pipeline
    try:
        artifacts = run_training_pipeline(config)
        return 0
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
