"""
Traffic Congestion Prediction Model Trainer
Version: 2.1

Features:
- Hyperparameter tuning with Bayesian optimization
- Comprehensive model evaluation
- Feature importance analysis
- Model versioning and metadata tracking
- Data validation
- Production-ready logging
"""

# ========================== Imports ==========================
import sys
import json
import joblib
import logging
import pandas as pd
import numpy as np

from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
)

from skopt import BayesSearchCV
from skopt.space import Integer, Real

# ========================== Configuration ==========================
MODEL_DIR = Path("../models")
LOG_DIR = Path("../logs")
DATA_PATH = Path("../data/processed_data/processed_traffic_data.csv")

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# ========================== Logging Setup ==========================
def setup_logging():
    LOG_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"model_training_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# ========================== Data Validation ==========================
class DataValidator:
    @staticmethod
    def validate_traffic_data(df: pd.DataFrame) -> pd.DataFrame:
        """Validate traffic data meets expected schema and constraints"""
        required_cols = ['traffic_volume', 'hour', 'day_of_week', 'latitude', 'longitude']
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            raise ValueError(f"Data missing required columns: {missing_cols}")

        if (df['traffic_volume'] < 0).any():
            logger.warning("Negative traffic values found. Clipping to zero.")
            df['traffic_volume'] = df['traffic_volume'].clip(lower=0)

        if not df['latitude'].between(-90, 90).all() or not df['longitude'].between(-180, 180).all():
            logger.error("Invalid coordinate values detected")
            raise ValueError("Latitude must be [-90, 90] and longitude [-180, 180]")

        return df

# ========================== Model Trainer ==========================
class ModelTrainer:
    def __init__(self, random_state: int = RANDOM_STATE):
        self.random_state = random_state
        self.bayes_search = None

    def prepare_data(self, data_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        logger.info(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)
        df = DataValidator.validate_traffic_data(df)

        X = df.drop(columns=["traffic_volume"])
        y = df["traffic_volume"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=self.random_state)
        return X_train, X_test, y_train, y_test

    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[RandomForestRegressor, Dict]:
        logger.info("Starting Bayesian hyperparameter search...")

        search_spaces = {
            'n_estimators': Integer(50, 300),
            'max_depth': Integer(3, 30),
            'min_samples_split': Integer(2, 20),
            'min_samples_leaf': Integer(1, 10),
            'max_features': Real(0.1, 1.0, prior='uniform')
        }

        model = RandomForestRegressor(random_state=self.random_state)

        self.bayes_search = BayesSearchCV(
            estimator=model,
            search_spaces=search_spaces,
            n_iter=30,
            cv=CV_FOLDS,
            n_jobs=-1,
            random_state=self.random_state
        )

        self.bayes_search.fit(X_train, y_train)
        logger.info("Hyperparameter search completed")

        return self.bayes_search.best_estimator_, self.bayes_search.best_params_

    def evaluate_model(
        self, model: RandomForestRegressor,
        X_test: pd.DataFrame, y_test: pd.Series,
        X_train: pd.DataFrame, y_train: pd.Series
    ) -> Dict:

        y_pred = model.predict(X_test)
        errors = y_test - y_pred

        metrics = {
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred),
            'explained_variance': explained_variance_score(y_test, y_pred),
            'error_stats': {
                'mean': float(np.mean(errors)),
                'std': float(np.std(errors)),
                'q25': float(np.percentile(errors, 25)),
                'q75': float(np.percentile(errors, 75)),
            },
            'feature_importance': dict(zip(X_test.columns, model.feature_importances_)),
        }

        cv_scores = cross_val_score(
            model,
            pd.concat([X_test, X_train]),
            pd.concat([y_test, y_train]),
            cv=CV_FOLDS,
            scoring='neg_mean_squared_error'
        )

        metrics['cv_scores'] = {
            'mean': float(np.mean(cv_scores)),
            'std': float(np.std(cv_scores)),
        }

        return metrics

# ========================== Model Saver ==========================
class ModelSaver:
    @staticmethod
    def save_model_artifacts(model, params, metrics, feature_names) -> Dict:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        MODEL_DIR.mkdir(exist_ok=True)

        model_path = MODEL_DIR / f"traffic_model_{timestamp}.pkl"
        joblib.dump(model, model_path)

        meta = {
            'model_version': timestamp,
            'model_type': 'RandomForestRegressor',
            'training_date': timestamp,
            'git_hash': None,
            'best_params': params,
            'metrics': metrics,
            'feature_names': feature_names,
            'python_version': sys.version,
            'dependencies': {
                'pandas': pd.__version__,
                'numpy': np.__version__,
                'sklearn': joblib.__version__
            }
        }

        meta_path = MODEL_DIR / f"model_meta_{timestamp}.json"
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)

        logger.info(f"Model saved to {model_path}")
        logger.info(f"Metadata saved to {meta_path}")

        return {
            'model_path': str(model_path),
            'meta_path': str(meta_path)
        }

# ========================== Pipeline Entry ==========================
def main():
    try:
        logger.info("Starting model training pipeline")

        trainer = ModelTrainer()
        saver = ModelSaver()

        X_train, X_test, y_train, y_test = trainer.prepare_data(DATA_PATH)
        model, best_params = trainer.train_model(X_train, y_train)
        metrics = trainer.evaluate_model(model, X_test, y_test, X_train, y_train)

        artifacts = saver.save_model_artifacts(
            model=model,
            params=best_params,
            metrics=metrics,
            feature_names=list(X_train.columns)
        )

        logger.info("Model training completed successfully")
        logger.info(f"Final RMSE: {metrics['rmse']:.2f}")
        logger.info(f"R² Score: {metrics['r2']:.2f}")

        return artifacts

    except Exception as e:
        logger.error(f"Model training failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
