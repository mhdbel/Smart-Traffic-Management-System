# utils/data_preprocessing.py
"""
Data preprocessing utilities for the Smart Traffic Management System.

This module provides functions for cleaning, transforming, and preparing
traffic data for machine learning models.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Configure logging
logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS
# =============================================================================

REQUIRED_COLUMNS = ['date_time', 'traffic_volume']
OPTIONAL_COLUMNS = ['weather_main', 'temperature', 'humidity', 'wind_speed', 'location']
CATEGORICAL_COLUMNS = ['weather_main', 'location', 'temp_category']
DATETIME_FORMATS = ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d', '%d/%m/%Y %H:%M', '%m/%d/%Y %H:%M']


# =============================================================================
# EXCEPTIONS
# =============================================================================

class DataPreprocessingError(Exception):
    """Base exception for preprocessing errors."""
    pass


class DataNotFoundError(DataPreprocessingError):
    """Raised when input data file is not found."""
    pass


class DataValidationError(DataPreprocessingError):
    """Raised when data validation fails."""
    pass


class ConfigurationError(DataPreprocessingError):
    """Raised when configuration is invalid."""
    pass


# =============================================================================
# DATA LOADING AND VALIDATION
# =============================================================================

def load_data(
    file_path: Union[str, Path],
    date_columns: Optional[List[str]] = None,
    encoding: str = 'utf-8'
) -> pd.DataFrame:
    """
    Load data from CSV file with error handling.
    
    Args:
        file_path: Path to CSV file
        date_columns: Columns to parse as dates
        encoding: File encoding
        
    Returns:
        Loaded DataFrame
        
    Raises:
        DataNotFoundError: If file doesn't exist
        DataValidationError: If file is invalid
    """
    path = Path(file_path)
    
    if not path.exists():
        raise DataNotFoundError(f"Data file not found: {path}")
    
    if not path.suffix.lower() == '.csv':
        logger.warning(f"File extension is not .csv: {path.suffix}")
    
    try:
        df = pd.read_csv(
            path,
            encoding=encoding,
            parse_dates=date_columns,
        )
        
        if df.empty:
            raise DataValidationError("Loaded DataFrame is empty")
        
        logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns from {path}")
        return df
        
    except pd.errors.EmptyDataError:
        raise DataValidationError(f"File is empty: {path}")
    except pd.errors.ParserError as e:
        raise DataValidationError(f"Error parsing CSV: {e}")
    except UnicodeDecodeError as e:
        raise DataValidationError(f"Encoding error (try different encoding): {e}")


def validate_data(
    df: pd.DataFrame,
    required_columns: Optional[List[str]] = None,
    min_rows: int = 10
) -> None:
    """
    Validate DataFrame structure and content.
    
    Args:
        df: DataFrame to validate
        required_columns: Required column names
        min_rows: Minimum required rows
        
    Raises:
        DataValidationError: If validation fails
    """
    required = required_columns or REQUIRED_COLUMNS
    
    # Check required columns
    missing = set(required) - set(df.columns)
    if missing:
        raise DataValidationError(f"Missing required columns: {missing}")
    
    # Check minimum rows
    if len(df) < min_rows:
        raise DataValidationError(
            f"Insufficient data: {len(df)} rows (minimum: {min_rows})"
        )
    
    # Check for completely empty columns
    empty_cols = [col for col in df.columns if df[col].isna().all()]
    if empty_cols:
        logger.warning(f"Completely empty columns: {empty_cols}")
    
    # Check traffic_volume is numeric
    if 'traffic_volume' in df.columns:
        if not pd.api.types.is_numeric_dtype(df['traffic_volume']):
            raise DataValidationError("traffic_volume must be numeric")
        
        if (df['traffic_volume'] < 0).any():
            logger.warning("Negative traffic_volume values found")
    
    logger.info("Data validation passed")


# =============================================================================
# DATETIME PROCESSING
# =============================================================================

def parse_datetime(
    df: pd.DataFrame,
    datetime_column: str = 'date_time',
    formats: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Parse datetime column and extract time features.
    
    Args:
        df: Input DataFrame
        datetime_column: Name of datetime column
        formats: List of datetime formats to try
        
    Returns:
        DataFrame with parsed dates and time features
    """
    df = df.copy()
    formats = formats or DATETIME_FORMATS
    
    if datetime_column not in df.columns:
        logger.warning(f"Datetime column '{datetime_column}' not found")
        return df
    
    # Try to parse datetime
    if not pd.api.types.is_datetime64_any_dtype(df[datetime_column]):
        parsed = False
        for fmt in formats:
            try:
                df[datetime_column] = pd.to_datetime(
                    df[datetime_column], format=fmt, errors='raise'
                )
                logger.info(f"Parsed datetime using format: {fmt}")
                parsed = True
                break
            except (ValueError, TypeError):
                continue
        
        if not parsed:
            df[datetime_column] = pd.to_datetime(
                df[datetime_column], errors='coerce'
            )
            logger.info("Parsed datetime using automatic inference")
    
    # Remove invalid dates
    invalid_count = df[datetime_column].isna().sum()
    if invalid_count > 0:
        logger.warning(f"Removing {invalid_count} rows with invalid dates")
        df = df.dropna(subset=[datetime_column])
    
    # Extract time features
    dt = df[datetime_column].dt
    
    df['hour'] = dt.hour
    df['day_of_week'] = dt.dayofweek
    df['day_of_month'] = dt.day
    df['month'] = dt.month
    df['year'] = dt.year
    df['week_of_year'] = dt.isocalendar().week.astype(int)
    df['quarter'] = dt.quarter
    
    # Binary features
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['is_morning_rush'] = df['hour'].between(7, 9).astype(int)
    df['is_evening_rush'] = df['hour'].between(16, 19).astype(int)
    df['is_rush_hour'] = ((df['is_morning_rush'] == 1) | (df['is_evening_rush'] == 1)).astype(int)
    df['is_night'] = df['hour'].between(22, 6).astype(int)
    
    # Cyclical encoding for hour and day
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    logger.info(f"Extracted {12} datetime features")
    
    return df


# =============================================================================
# MISSING VALUE HANDLING
# =============================================================================

def handle_missing_values(
    df: pd.DataFrame,
    strategy: str = 'median',
    fill_values: Optional[Dict[str, any]] = None
) -> pd.DataFrame:
    """
    Handle missing values in the dataset.
    
    Args:
        df: Input DataFrame
        strategy: 'median', 'mean', 'mode', or 'drop'
        fill_values: Custom fill values per column
        
    Returns:
        DataFrame with missing values handled
    """
    df = df.copy()
    fill_values = fill_values or {}
    
    initial_missing = df.isnull().sum().sum()
    
    if initial_missing == 0:
        logger.info("No missing values found")
        return df
    
    # Apply custom fill values first
    for col, value in fill_values.items():
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna(value)
            logger.debug(f"Filled {col} with custom value: {value}")
    
    # Handle remaining missing values by data type
    numeric_cols = df.select_dtypes(include=['number']).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    for col in numeric_cols:
        if df[col].isnull().any():
            if strategy == 'median':
                fill_val = df[col].median()
            elif strategy == 'mean':
                fill_val = df[col].mean()
            elif strategy == 'mode':
                fill_val = df[col].mode().iloc[0] if not df[col].mode().empty else 0
            elif strategy == 'drop':
                continue  # Handle below
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            df[col] = df[col].fillna(fill_val)
            logger.debug(f"Filled {col} with {strategy}: {fill_val:.2f}")
    
    for col in categorical_cols:
        if df[col].isnull().any():
            if strategy == 'drop':
                continue
            mode_val = df[col].mode().iloc[0] if not df[col].mode().empty else 'Unknown'
            df[col] = df[col].fillna(mode_val)
            logger.debug(f"Filled {col} with mode: {mode_val}")
    
    if strategy == 'drop':
        df = df.dropna()
    
    final_missing = df.isnull().sum().sum()
    logger.info(f"Missing values: {initial_missing} -> {final_missing}")
    
    return df


# =============================================================================
# OUTLIER HANDLING
# =============================================================================

def handle_outliers(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    method: str = 'iqr',
    threshold: float = 1.5,
    strategy: str = 'cap'
) -> pd.DataFrame:
    """
    Detect and handle outliers in numeric columns.
    
    Args:
        df: Input DataFrame
        columns: Columns to check (default: all numeric)
        method: 'iqr' or 'zscore'
        threshold: IQR multiplier or z-score threshold
        strategy: 'cap', 'remove', or 'nan'
        
    Returns:
        DataFrame with outliers handled
    """
    df = df.copy()
    
    if columns is None:
        columns = df.select_dtypes(include=['number']).columns.tolist()
        # Exclude binary columns
        columns = [c for c in columns if df[c].nunique() > 2]
    
    total_outliers = 0
    
    for col in columns:
        if col not in df.columns:
            continue
        
        if method == 'iqr':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - threshold * IQR
            upper = Q3 + threshold * IQR
            outliers_mask = (df[col] < lower) | (df[col] > upper)
            
        elif method == 'zscore':
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            outliers_mask = z_scores > threshold
            lower = df[col].mean() - threshold * df[col].std()
            upper = df[col].mean() + threshold * df[col].std()
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        n_outliers = outliers_mask.sum()
        if n_outliers == 0:
            continue
        
        total_outliers += n_outliers
        
        if strategy == 'cap':
            df.loc[df[col] < lower, col] = lower
            df.loc[df[col] > upper, col] = upper
        elif strategy == 'remove':
            df = df[~outliers_mask]
        elif strategy == 'nan':
            df.loc[outliers_mask, col] = np.nan
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        logger.debug(f"Handled {n_outliers} outliers in {col} ({strategy})")
    
    logger.info(f"Total outliers handled: {total_outliers}")
    
    return df


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived features for better ML performance.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with engineered features
    """
    df = df.copy()
    initial_cols = len(df.columns)
    
    # Traffic-based features
    if 'traffic_volume' in df.columns:
        # Log transform
        df['traffic_volume_log'] = np.log1p(df['traffic_volume'])
        
        # Lag features (if sorted by time)
        if 'date_time' in df.columns:
            df = df.sort_values('date_time')
            df['traffic_lag_1h'] = df['traffic_volume'].shift(1)
            df['traffic_lag_24h'] = df['traffic_volume'].shift(24)
            
            # Rolling statistics
            df['traffic_rolling_mean_3h'] = df['traffic_volume'].rolling(3, min_periods=1).mean()
            df['traffic_rolling_std_3h'] = df['traffic_volume'].rolling(3, min_periods=1).std().fillna(0)
            df['traffic_rolling_mean_24h'] = df['traffic_volume'].rolling(24, min_periods=1).mean()
        
        # Percentile rank
        df['traffic_percentile'] = df['traffic_volume'].rank(pct=True)
    
    # Weather features
    if 'temperature' in df.columns:
        # Temperature categories
        df['temp_category'] = pd.cut(
            df['temperature'],
            bins=[-np.inf, 0, 10, 20, 30, np.inf],
            labels=['freezing', 'cold', 'mild', 'warm', 'hot']
        )
        
        # Extreme temperature
        df['is_extreme_temp'] = (
            (df['temperature'] < 0) | (df['temperature'] > 35)
        ).astype(int)
    
    if 'humidity' in df.columns:
        df['is_high_humidity'] = (df['humidity'] > 80).astype(int)
    
    if 'wind_speed' in df.columns:
        df['is_high_wind'] = (df['wind_speed'] > 10).astype(int)
    
    # Interaction features
    if 'is_rush_hour' in df.columns and 'is_weekend' in df.columns:
        df['rush_hour_weekday'] = (
            (df['is_rush_hour'] == 1) & (df['is_weekend'] == 0)
        ).astype(int)
    
    new_cols = len(df.columns) - initial_cols
    logger.info(f"Created {new_cols} new features. Total: {len(df.columns)}")
    
    return df


# =============================================================================
# ENCODING
# =============================================================================

def encode_categorical(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    method: str = 'onehot',
    drop_first: bool = True,
    max_categories: int = 10
) -> pd.DataFrame:
    """
    Encode categorical variables.
    
    Args:
        df: Input DataFrame
        columns: Columns to encode (default: detect automatically)
        method: 'onehot' or 'label'
        drop_first: Drop first category for one-hot
        max_categories: Max categories for one-hot (use label for more)
        
    Returns:
        DataFrame with encoded categorical variables
    """
    df = df.copy()
    
    if columns is None:
        columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        columns = [c for c in columns if c in CATEGORICAL_COLUMNS or df[c].nunique() <= max_categories]
    
    if not columns:
        logger.info("No categorical columns to encode")
        return df
    
    for col in columns:
        if col not in df.columns:
            continue
        
        n_categories = df[col].nunique()
        
        if method == 'onehot' and n_categories <= max_categories:
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=drop_first)
            df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
            logger.debug(f"One-hot encoded {col} ({n_categories} categories)")
            
        elif method == 'label' or n_categories > max_categories:
            df[f'{col}_encoded'] = df[col].astype('category').cat.codes
            logger.debug(f"Label encoded {col} ({n_categories} categories)")
    
    logger.info(f"Encoded {len(columns)} categorical columns")
    
    return df


# =============================================================================
# SCALING
# =============================================================================

def scale_features(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    method: str = 'standard',
    exclude: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, Dict]:
    """
    Scale numeric features.
    
    Args:
        df: Input DataFrame
        columns: Columns to scale (default: all numeric)
        method: 'standard', 'minmax', or 'robust'
        exclude: Columns to exclude from scaling
        
    Returns:
        Tuple of (scaled DataFrame, scaling parameters)
    """
    df = df.copy()
    exclude = exclude or ['traffic_volume']  # Don't scale target by default
    
    if columns is None:
        columns = df.select_dtypes(include=['number']).columns.tolist()
    
    columns = [c for c in columns if c not in exclude and c in df.columns]
    
    scalers = {}
    
    for col in columns:
        if df[col].std() == 0:
            logger.warning(f"Skipping {col}: zero variance")
            continue
        
        if method == 'standard':
            mean = df[col].mean()
            std = df[col].std()
            df[col] = (df[col] - mean) / std
            scalers[col] = {'method': 'standard', 'mean': mean, 'std': std}
            
        elif method == 'minmax':
            min_val = df[col].min()
            max_val = df[col].max()
            df[col] = (df[col] - min_val) / (max_val - min_val)
            scalers[col] = {'method': 'minmax', 'min': min_val, 'max': max_val}
            
        elif method == 'robust':
            median = df[col].median()
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            df[col] = (df[col] - median) / iqr if iqr != 0 else 0
            scalers[col] = {'method': 'robust', 'median': median, 'iqr': iqr}
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    logger.info(f"Scaled {len(scalers)} columns using {method}")
    
    return df, scalers


# =============================================================================
# DATA SPLITTING
# =============================================================================

def split_data(
    df: pd.DataFrame,
    target_column: str = 'traffic_volume',
    test_size: float = 0.2,
    val_size: float = 0.1,
    temporal: bool = True,
    random_state: int = 42
) -> Dict[str, pd.DataFrame]:
    """
    Split data into train/validation/test sets.
    
    Args:
        df: Input DataFrame
        target_column: Target variable column
        test_size: Proportion for test set
        val_size: Proportion for validation set
        temporal: Use temporal split (recommended for time series)
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary with train/val/test DataFrames
    """
    from sklearn.model_selection import train_test_split
    
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found")
    
    if temporal and 'date_time' in df.columns:
        df = df.sort_values('date_time').reset_index(drop=True)
        
        n = len(df)
        train_end = int(n * (1 - test_size - val_size))
        val_end = int(n * (1 - test_size))
        
        splits = {
            'train': df.iloc[:train_end].copy(),
            'val': df.iloc[train_end:val_end].copy(),
            'test': df.iloc[val_end:].copy(),
        }
        
        logger.info("Using temporal split")
    else:
        train_df, temp_df = train_test_split(
            df, test_size=test_size + val_size, random_state=random_state
        )
        
        val_ratio = val_size / (test_size + val_size)
        val_df, test_df = train_test_split(
            temp_df, test_size=1-val_ratio, random_state=random_state
        )
        
        splits = {
            'train': train_df.reset_index(drop=True),
            'val': val_df.reset_index(drop=True),
            'test': test_df.reset_index(drop=True),
        }
        
        logger.info("Using random split")
    
    for name, data in splits.items():
        logger.info(f"{name}: {len(data)} rows ({len(data)/len(df)*100:.1f}%)")
    
    return splits


# =============================================================================
# QUALITY REPORT
# =============================================================================

def generate_quality_report(
    df: pd.DataFrame,
    output_path: Optional[str] = None
) -> Dict:
    """
    Generate a comprehensive data quality report.
    
    Args:
        df: DataFrame to analyze
        output_path: Optional path to save JSON report
        
    Returns:
        Quality report dictionary
    """
    report = {
        'generated_at': datetime.now().isoformat(),
        'shape': {
            'rows': len(df),
            'columns': len(df.columns),
        },
        'memory_usage_mb': round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
        'duplicates': int(df.duplicated().sum()),
        'columns': {},
        'missing_summary': {},
        'data_types': df.dtypes.astype(str).to_dict(),
    }
    
    for col in df.columns:
        col_info = {
            'dtype': str(df[col].dtype),
            'missing': int(df[col].isnull().sum()),
            'missing_pct': round(df[col].isnull().sum() / len(df) * 100, 2),
            'unique': int(df[col].nunique()),
            'unique_pct': round(df[col].nunique() / len(df) * 100, 2),
        }
        
        if pd.api.types.is_numeric_dtype(df[col]):
            non_null = df[col].dropna()
            if len(non_null) > 0:
                col_info.update({
                    'min': float(non_null.min()),
                    'max': float(non_null.max()),
                    'mean': float(non_null.mean()),
                    'median': float(non_null.median()),
                    'std': float(non_null.std()),
                    'skew': float(non_null.skew()),
                    'zeros': int((non_null == 0).sum()),
                    'negatives': int((non_null < 0).sum()),
                })
        
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            non_null = df[col].dropna()
            if len(non_null) > 0:
                col_info.update({
                    'min': str(non_null.min()),
                    'max': str(non_null.max()),
                    'range_days': (non_null.max() - non_null.min()).days,
                })
        
        else:
            # Categorical
            if df[col].nunique() <= 20:
                col_info['top_values'] = df[col].value_counts().head(10).to_dict()
        
        report['columns'][col] = col_info
        
        if col_info['missing'] > 0:
            report['missing_summary'][col] = {
                'count': col_info['missing'],
                'percentage': col_info['missing_pct'],
            }
    
    # Summary
    report['summary'] = {
        'total_missing_values': sum(c['missing'] for c in report['columns'].values()),
        'columns_with_missing': len(report['missing_summary']),
        'numeric_columns': len(df.select_dtypes(include=['number']).columns),
        'categorical_columns': len(df.select_dtypes(include=['object', 'category']).columns),
        'datetime_columns': len(df.select_dtypes(include=['datetime64']).columns),
        'high_cardinality_columns': [
            col for col, info in report['columns'].items()
            if info['unique_pct'] > 90
        ],
    }
    
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"Saved quality report to {path}")
    
    return report


# =============================================================================
# PREPROCESSING PIPELINE
# =============================================================================

@dataclass
class PreprocessingStep:
    """Single preprocessing step configuration."""
    name: str
    func: Callable[[pd.DataFrame], pd.DataFrame]
    enabled: bool = True
    kwargs: Dict = field(default_factory=dict)


class PreprocessingPipeline:
    """
    Configurable, reproducible preprocessing pipeline.
    
    Example:
        pipeline = PreprocessingPipeline()
        pipeline.add_step("validate", validate_data)
        pipeline.add_step("parse_dates", parse_datetime)
        pipeline.add_step("handle_missing", handle_missing_values, strategy='median')
        pipeline.add_step("handle_outliers", handle_outliers, method='iqr')
        pipeline.add_step("engineer_features", engineer_features)
        pipeline.add_step("encode", encode_categorical)
        
        result = pipeline.run(df)
        pipeline.save_config("pipeline_config.json")
    """
    
    def __init__(self, name: str = "default"):
        self.name = name
        self.steps: List[PreprocessingStep] = []
        self.run_history: List[Dict] = []
    
    def add_step(
        self,
        name: str,
        func: Callable,
        enabled: bool = True,
        **kwargs
    ) -> 'PreprocessingPipeline':
        """Add a preprocessing step."""
        self.steps.append(PreprocessingStep(name, func, enabled, kwargs))
        return self
    
    def remove_step(self, name: str) -> 'PreprocessingPipeline':
        """Remove a step by name."""
        self.steps = [s for s in self.steps if s.name != name]
        return self
    
    def enable_step(self, name: str) -> 'PreprocessingPipeline':
        """Enable a step by name."""
        for step in self.steps:
            if step.name == name:
                step.enabled = True
        return self
    
    def disable_step(self, name: str) -> 'PreprocessingPipeline':
        """Disable a step by name."""
        for step in self.steps:
            if step.name == name:
                step.enabled = False
        return self
    
    def run(self, df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
        """Execute all enabled steps in order."""
        result = df.copy()
        run_info = {
            'started_at': datetime.now().isoformat(),
            'input_shape': df.shape,
            'steps_executed': [],
        }
        
        for step in self.steps:
            if not step.enabled:
                if verbose:
                    logger.info(f"⏭️ Skipping: {step.name}")
                continue
            
            if verbose:
                logger.info(f"▶️ Running: {step.name}")
            
            start_time = datetime.now()
            try:
                if step.kwargs:
                    result = step.func(result, **step.kwargs)
                else:
                    result = step.func(result)
                
                duration = (datetime.now() - start_time).total_seconds()
                run_info['steps_executed'].append({
                    'name': step.name,
                    'success': True,
                    'duration_seconds': duration,
                    'output_shape': result.shape,
                })
                
                if verbose:
                    logger.info(f"✅ {step.name}: {result.shape} ({duration:.2f}s)")
                    
            except Exception as e:
                logger.error(f"❌ Error in {step.name}: {e}")
                run_info['steps_executed'].append({
                    'name': step.name,
                    'success': False,
                    'error': str(e),
                })
                raise
        
        run_info['completed_at'] = datetime.now().isoformat()
        run_info['output_shape'] = result.shape
        self.run_history.append(run_info)
        
        return result
    
    def get_config(self) -> Dict:
        """Get pipeline configuration."""
        return {
            'name': self.name,
            'steps': [
                {
                    'name': s.name,
                    'enabled': s.enabled,
                    'kwargs': s.kwargs,
                }
                for s in self.steps
            ],
        }
    
    def save_config(self, path: str) -> None:
        """Save pipeline configuration to JSON."""
        with open(path, 'w') as f:
            json.dump(self.get_config(), f, indent=2)
        logger.info(f"Saved pipeline config to {path}")


# =============================================================================
# MAIN PREPROCESSING FUNCTION
# =============================================================================

def preprocess_data(
    raw_file: Union[str, Path],
    processed_file: Union[str, Path],
    config: Optional[Dict] = None,
    generate_report: bool = True
) -> pd.DataFrame:
    """
    Main preprocessing function - runs complete pipeline.
    
    Args:
        raw_file: Path to raw data file
        processed_file: Path to save processed data
        config: Optional configuration dictionary
        generate_report: Whether to generate quality report
        
    Returns:
        Processed DataFrame
    """
    config = config or {}
    
    # Load data
    logger.info(f"Loading data from {raw_file}")
    df = load_data(raw_file)
    
    # Validate
    validate_data(df)
    
    # Generate initial quality report
    if generate_report:
        report_path = Path(processed_file).parent / 'quality_report_raw.json'
        generate_quality_report(df, str(report_path))
    
    # Create and run pipeline
    pipeline = PreprocessingPipeline("traffic_preprocessing")
    
    pipeline.add_step("parse_datetime", parse_datetime)
    pipeline.add_step(
        "handle_missing",
        handle_missing_values,
        strategy=config.get('missing_strategy', 'median')
    )
    pipeline.add_step(
        "handle_outliers",
        handle_outliers,
        method=config.get('outlier_method', 'iqr'),
        threshold=config.get('outlier_threshold', 1.5)
    )
    pipeline.add_step("engineer_features", engineer_features)
    pipeline.add_step(
        "encode_categorical",
        encode_categorical,
        method=config.get('encoding_method', 'onehot')
    )
    
    # Run pipeline
    df = pipeline.run(df)
    
    # Save processed data
    output_path = Path(processed_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved processed data to {output_path}")
    
    # Generate final quality report
    if generate_report:
        report_path = Path(processed_file).parent / 'quality_report_processed.json'
        generate_quality_report(df, str(report_path))
    
    # Save pipeline config
    config_path = Path(processed_file).parent / 'pipeline_config.json'
    pipeline.save_config(str(config_path))
    
    return df


# =============================================================================
# CLI
# =============================================================================

def main():
    """Command-line interface for data preprocessing."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Preprocess traffic data for ML training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python data_preprocessing.py -i data/raw/traffic.csv -o data/processed/traffic.csv
  python data_preprocessing.py -i data/raw/traffic.csv -o data/processed/traffic.csv --missing-strategy mean
  python data_preprocessing.py -i data/raw/traffic.csv -o data/processed/traffic.csv -v --no-report
        """
    )
    
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Path to raw data CSV file"
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Path to save processed data"
    )
    parser.add_argument(
        "--missing-strategy",
        choices=['median', 'mean', 'mode', 'drop'],
        default='median',
        help="Strategy for handling missing values (default: median)"
    )
    parser.add_argument(
        "--outlier-method",
        choices=['iqr', 'zscore'],
        default='iqr',
        help="Method for detecting outliers (default: iqr)"
    )
    parser.add_argument(
        "--outlier-threshold",
        type=float,
        default=1.5,
        help="Threshold for outlier detection (default: 1.5)"
    )
    parser.add_argument(
        "--encoding-method",
        choices=['onehot', 'label'],
        default='onehot',
        help="Method for encoding categorical variables (default: onehot)"
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Skip generating quality reports"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Build config
    config = {
        'missing_strategy': args.missing_strategy,
        'outlier_method': args.outlier_method,
        'outlier_threshold': args.outlier_threshold,
        'encoding_method': args.encoding_method,
    }
    
    try:
        df = preprocess_data(
            args.input,
            args.output,
            config=config,
            generate_report=not args.no_report
        )
        logger.info(f"✅ Preprocessing complete! Output: {args.output}")
        logger.info(f"   Shape: {df.shape[0]} rows, {df.shape[1]} columns")
        
    except DataPreprocessingError as e:
        logger.error(f"❌ Preprocessing failed: {e}")
        exit(1)
    except Exception as e:
        logger.exception(f"❌ Unexpected error: {e}")
        exit(1)


if __name__ == "__main__":
    main()
