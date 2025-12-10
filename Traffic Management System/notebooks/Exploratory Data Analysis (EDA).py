# notebooks/exploratory_data_analysis.py
"""
Exploratory Data Analysis (EDA) for Smart Traffic Management System.

This notebook provides comprehensive analysis of traffic data including:
1. Data Overview and Quality Assessment
2. Temporal Analysis
3. Distribution Analysis
4. Correlation Analysis
5. Feature Engineering Insights
6. Anomaly Detection
7. Actionable Insights

Author: Traffic Management Team
Last Updated: 2024
"""

# =============================================================================
# IMPORTS
# =============================================================================

import os
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Configure pandas display
pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 100)
pd.set_option('display.float_format', '{:.2f}'.format)

# Figure settings
FIGSIZE_SMALL = (10, 6)
FIGSIZE_MEDIUM = (12, 8)
FIGSIZE_LARGE = (16, 10)
FIGSIZE_WIDE = (16, 6)
DPI = 100

# Colors
COLORS = {
    'primary': '#3498db',
    'secondary': '#2ecc71',
    'warning': '#f39c12',
    'danger': '#e74c3c',
    'info': '#9b59b6',
}


# =============================================================================
# CONFIGURATION
# =============================================================================

class EDAConfig:
    """Configuration for EDA notebook."""
    
    # Paths
    BASE_DIR = Path(__file__).parent.parent if '__file__' in dir() else Path('.')
    DATA_DIR = BASE_DIR / "data"
    RAW_DATA_PATH = DATA_DIR / "raw_data" / "traffic_data.csv"
    PROCESSED_DATA_PATH = DATA_DIR / "processed_data" / "processed_traffic_data.csv"
    OUTPUT_DIR = BASE_DIR / "reports" / "eda"
    
    # Column names
    DATETIME_COL = 'date_time'
    TARGET_COL = 'traffic_volume'
    
    # Analysis parameters
    OUTLIER_THRESHOLD = 3.0  # Z-score threshold
    CORRELATION_THRESHOLD = 0.5  # For highlighting strong correlations
    
    @classmethod
    def get_data_path(cls) -> Path:
        """Get the data path, checking both processed and raw."""
        if cls.PROCESSED_DATA_PATH.exists():
            return cls.PROCESSED_DATA_PATH
        elif cls.RAW_DATA_PATH.exists():
            return cls.RAW_DATA_PATH
        else:
            # Try environment variable
            env_path = os.getenv('TRAFFIC_DATA_PATH')
            if env_path and Path(env_path).exists():
                return Path(env_path)
            raise FileNotFoundError(
                f"Data file not found. Checked:\n"
                f"  - {cls.PROCESSED_DATA_PATH}\n"
                f"  - {cls.RAW_DATA_PATH}\n"
                f"  - TRAFFIC_DATA_PATH env variable"
            )


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data(filepath: Optional[Path] = None) -> pd.DataFrame:
    """
    Load traffic data with proper parsing.
    
    Args:
        filepath: Optional path to data file
        
    Returns:
        Loaded DataFrame
    """
    filepath = filepath or EDAConfig.get_data_path()
    
    print(f"📂 Loading data from: {filepath}")
    
    # Load data
    df = pd.read_csv(filepath)
    
    # Parse datetime
    datetime_col = EDAConfig.DATETIME_COL
    if datetime_col in df.columns:
        df[datetime_col] = pd.to_datetime(df[datetime_col], errors='coerce')
        df = df.sort_values(datetime_col).reset_index(drop=True)
    
    print(f"✅ Loaded {len(df):,} rows, {len(df.columns)} columns")
    
    return df


# =============================================================================
# 1. DATA OVERVIEW
# =============================================================================

def print_section(title: str) -> None:
    """Print a section header."""
    print(f"\n{'='*60}")
    print(f"📊 {title}")
    print('='*60)


def data_overview(df: pd.DataFrame) -> Dict:
    """
    Provide comprehensive data overview.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary with overview statistics
    """
    print_section("1. DATA OVERVIEW")
    
    # Basic info
    print(f"\n📋 Basic Information:")
    print(f"   • Rows: {len(df):,}")
    print(f"   • Columns: {len(df.columns)}")
    print(f"   • Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Date range
    datetime_col = EDAConfig.DATETIME_COL
    if datetime_col in df.columns and pd.api.types.is_datetime64_any_dtype(df[datetime_col]):
        date_range = df[datetime_col].max() - df[datetime_col].min()
        print(f"\n📅 Date Range:")
        print(f"   • Start: {df[datetime_col].min()}")
        print(f"   • End: {df[datetime_col].max()}")
        print(f"   • Duration: {date_range.days} days")
    
    # Column types
    print(f"\n📑 Column Types:")
    dtype_counts = df.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        print(f"   • {dtype}: {count} columns")
    
    # Display columns
    print(f"\n📝 Columns:")
    for i, col in enumerate(df.columns, 1):
        dtype = df[col].dtype
        null_pct = df[col].isnull().sum() / len(df) * 100
        unique = df[col].nunique()
        print(f"   {i:2}. {col:<30} | {str(dtype):<15} | {null_pct:5.1f}% null | {unique:,} unique")
    
    # First few rows
    print(f"\n🔍 Sample Data (first 5 rows):")
    display(df.head()) if 'display' in dir() else print(df.head().to_string())
    
    # Statistics for numeric columns
    print(f"\n📈 Numeric Summary:")
    numeric_df = df.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        display(numeric_df.describe().T) if 'display' in dir() else print(numeric_df.describe().T.to_string())
    
    return {
        'n_rows': len(df),
        'n_columns': len(df.columns),
        'memory_mb': df.memory_usage(deep=True).sum() / 1024**2,
        'dtypes': dtype_counts.to_dict(),
    }


# =============================================================================
# 2. MISSING VALUE ANALYSIS
# =============================================================================

def missing_value_analysis(df: pd.DataFrame, plot: bool = True) -> pd.DataFrame:
    """
    Analyze missing values in the dataset.
    
    Args:
        df: Input DataFrame
        plot: Whether to create visualizations
        
    Returns:
        DataFrame with missing value statistics
    """
    print_section("2. MISSING VALUE ANALYSIS")
    
    # Calculate missing values
    missing = pd.DataFrame({
        'column': df.columns,
        'missing_count': df.isnull().sum().values,
        'missing_pct': (df.isnull().sum() / len(df) * 100).values,
        'dtype': df.dtypes.values
    })
    missing = missing.sort_values('missing_pct', ascending=False)
    
    # Summary
    total_missing = df.isnull().sum().sum()
    total_cells = df.size
    
    print(f"\n📊 Missing Value Summary:")
    print(f"   • Total missing cells: {total_missing:,} / {total_cells:,} ({total_missing/total_cells*100:.2f}%)")
    print(f"   • Columns with missing: {(missing['missing_count'] > 0).sum()} / {len(df.columns)}")
    
    # Columns with missing values
    cols_with_missing = missing[missing['missing_count'] > 0]
    if len(cols_with_missing) > 0:
        print(f"\n⚠️ Columns with Missing Values:")
        for _, row in cols_with_missing.iterrows():
            print(f"   • {row['column']}: {row['missing_count']:,} ({row['missing_pct']:.1f}%)")
    else:
        print(f"\n✅ No missing values found!")
    
    # Visualization
    if plot and len(cols_with_missing) > 0:
        fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)
        
        # Bar chart
        ax1 = axes[0]
        cols_to_plot = cols_with_missing.head(15)
        ax1.barh(cols_to_plot['column'], cols_to_plot['missing_pct'], color=COLORS['warning'])
        ax1.set_xlabel('Missing %')
        ax1.set_title('Missing Values by Column')
        ax1.invert_yaxis()
        
        # Heatmap of missing pattern
        ax2 = axes[1]
        missing_matrix = df.isnull().astype(int)
        if len(missing_matrix.columns) > 20:
            missing_matrix = missing_matrix[cols_with_missing['column'].head(20).tolist()]
        
        # Sample rows for visualization
        sample_idx = np.linspace(0, len(df)-1, min(100, len(df)), dtype=int)
        sns.heatmap(missing_matrix.iloc[sample_idx], cbar=False, cmap='Reds', ax=ax2)
        ax2.set_title('Missing Value Pattern (sampled rows)')
        ax2.set_ylabel('Row Index (sampled)')
        
        plt.tight_layout()
        plt.savefig(EDAConfig.OUTPUT_DIR / 'missing_values.png', dpi=DPI, bbox_inches='tight')
        plt.show()
    
    return missing


# =============================================================================
# 3. TARGET VARIABLE ANALYSIS
# =============================================================================

def target_analysis(df: pd.DataFrame, target_col: str = None) -> Dict:
    """
    Analyze the target variable (traffic_volume).
    
    Args:
        df: Input DataFrame
        target_col: Target column name
        
    Returns:
        Dictionary with target statistics
    """
    print_section("3. TARGET VARIABLE ANALYSIS")
    
    target_col = target_col or EDAConfig.TARGET_COL
    
    if target_col not in df.columns:
        print(f"⚠️ Target column '{target_col}' not found")
        return {}
    
    target = df[target_col].dropna()
    
    # Statistics
    stats_dict = {
        'count': len(target),
        'mean': target.mean(),
        'std': target.std(),
        'min': target.min(),
        'q1': target.quantile(0.25),
        'median': target.median(),
        'q3': target.quantile(0.75),
        'max': target.max(),
        'skewness': target.skew(),
        'kurtosis': target.kurtosis(),
    }
    
    print(f"\n📈 {target_col} Statistics:")
    for key, value in stats_dict.items():
        print(f"   • {key}: {value:,.2f}")
    
    # Distribution analysis
    print(f"\n📊 Distribution Analysis:")
    print(f"   • Skewness: {stats_dict['skewness']:.2f} ", end="")
    if abs(stats_dict['skewness']) < 0.5:
        print("(approximately symmetric)")
    elif stats_dict['skewness'] > 0:
        print("(right-skewed)")
    else:
        print("(left-skewed)")
    
    print(f"   • Kurtosis: {stats_dict['kurtosis']:.2f} ", end="")
    if stats_dict['kurtosis'] > 3:
        print("(heavy-tailed)")
    elif stats_dict['kurtosis'] < 3:
        print("(light-tailed)")
    else:
        print("(normal-like)")
    
    # Visualizations
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE_LARGE)
    
    # Distribution
    ax1 = axes[0, 0]
    sns.histplot(target, bins=50, kde=True, ax=ax1, color=COLORS['primary'])
    ax1.axvline(target.mean(), color='red', linestyle='--', label=f'Mean: {target.mean():.0f}')
    ax1.axvline(target.median(), color='green', linestyle='--', label=f'Median: {target.median():.0f}')
    ax1.set_title(f'Distribution of {target_col}')
    ax1.legend()
    
    # Box plot
    ax2 = axes[0, 1]
    sns.boxplot(x=target, ax=ax2, color=COLORS['primary'])
    ax2.set_title(f'Box Plot of {target_col}')
    
    # QQ plot
    ax3 = axes[1, 0]
    stats.probplot(target, dist="norm", plot=ax3)
    ax3.set_title('Q-Q Plot (Normal Distribution)')
    
    # Time series (if datetime available)
    ax4 = axes[1, 1]
    datetime_col = EDAConfig.DATETIME_COL
    if datetime_col in df.columns:
        df_sorted = df.sort_values(datetime_col)
        ax4.plot(df_sorted[datetime_col], df_sorted[target_col], alpha=0.5, linewidth=0.5)
        ax4.set_title(f'{target_col} Over Time')
        ax4.set_xlabel('Date')
        ax4.tick_params(axis='x', rotation=45)
    else:
        ax4.plot(target.values, alpha=0.5)
        ax4.set_title(f'{target_col} by Index')
    
    plt.tight_layout()
    plt.savefig(EDAConfig.OUTPUT_DIR / 'target_analysis.png', dpi=DPI, bbox_inches='tight')
    plt.show()
    
    return stats_dict


# =============================================================================
# 4. TEMPORAL ANALYSIS
# =============================================================================

def temporal_analysis(df: pd.DataFrame) -> Dict:
    """
    Analyze temporal patterns in traffic data.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary with temporal insights
    """
    print_section("4. TEMPORAL ANALYSIS")
    
    datetime_col = EDAConfig.DATETIME_COL
    target_col = EDAConfig.TARGET_COL
    
    if datetime_col not in df.columns:
        print(f"⚠️ Datetime column '{datetime_col}' not found")
        return {}
    
    # Ensure datetime type
    if not pd.api.types.is_datetime64_any_dtype(df[datetime_col]):
        df = df.copy()
        df[datetime_col] = pd.to_datetime(df[datetime_col], errors='coerce')
    
    # Extract time components
    df_temp = df.copy()
    df_temp['hour'] = df_temp[datetime_col].dt.hour
    df_temp['day_of_week'] = df_temp[datetime_col].dt.dayofweek
    df_temp['day_name'] = df_temp[datetime_col].dt.day_name()
    df_temp['month'] = df_temp[datetime_col].dt.month
    df_temp['month_name'] = df_temp[datetime_col].dt.month_name()
    df_temp['year'] = df_temp[datetime_col].dt.year
    df_temp['is_weekend'] = df_temp['day_of_week'].isin([5, 6])
    
    # Visualizations
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Hourly pattern
    ax1 = axes[0, 0]
    hourly = df_temp.groupby('hour')[target_col].mean()
    ax1.plot(hourly.index, hourly.values, marker='o', color=COLORS['primary'])
    ax1.fill_between(hourly.index, hourly.values, alpha=0.3)
    ax1.set_xlabel('Hour of Day')
    ax1.set_ylabel(f'Average {target_col}')
    ax1.set_title('📈 Hourly Traffic Pattern')
    ax1.set_xticks(range(0, 24, 2))
    
    # Highlight rush hours
    ax1.axvspan(7, 9, alpha=0.2, color='red', label='Morning Rush')
    ax1.axvspan(16, 19, alpha=0.2, color='orange', label='Evening Rush')
    ax1.legend()
    
    # 2. Daily pattern
    ax2 = axes[0, 1]
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    daily = df_temp.groupby('day_name')[target_col].mean().reindex(day_order)
    colors = ['#3498db'] * 5 + ['#2ecc71'] * 2
    ax2.bar(range(7), daily.values, color=colors)
    ax2.set_xticks(range(7))
    ax2.set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    ax2.set_ylabel(f'Average {target_col}')
    ax2.set_title('📅 Daily Traffic Pattern')
    
    # 3. Monthly pattern
    ax3 = axes[0, 2]
    monthly = df_temp.groupby('month')[target_col].mean()
    ax3.plot(monthly.index, monthly.values, marker='o', color=COLORS['secondary'])
    ax3.fill_between(monthly.index, monthly.values, alpha=0.3, color=COLORS['secondary'])
    ax3.set_xlabel('Month')
    ax3.set_ylabel(f'Average {target_col}')
    ax3.set_title('📆 Monthly Traffic Pattern')
    ax3.set_xticks(range(1, 13))
    
    # 4. Hour x Day heatmap
    ax4 = axes[1, 0]
    pivot = df_temp.pivot_table(
        values=target_col,
        index='day_name',
        columns='hour',
        aggfunc='mean'
    ).reindex(day_order)
    sns.heatmap(pivot, cmap='YlOrRd', ax=ax4, cbar_kws={'label': target_col})
    ax4.set_title('🗓️ Traffic Heatmap (Hour x Day)')
    
    # 5. Weekend vs Weekday
    ax5 = axes[1, 1]
    weekend_data = [
        df_temp[~df_temp['is_weekend']][target_col],
        df_temp[df_temp['is_weekend']][target_col]
    ]
    bp = ax5.boxplot(weekend_data, labels=['Weekday', 'Weekend'], patch_artist=True)
    bp['boxes'][0].set_facecolor(COLORS['primary'])
    bp['boxes'][1].set_facecolor(COLORS['secondary'])
    ax5.set_ylabel(target_col)
    ax5.set_title('📊 Weekday vs Weekend')
    
    # 6. Trend over time
    ax6 = axes[1, 2]
    if 'year' in df_temp.columns and df_temp['year'].nunique() > 1:
        yearly = df_temp.groupby('year')[target_col].mean()
        ax6.bar(yearly.index, yearly.values, color=COLORS['info'])
        ax6.set_xlabel('Year')
        ax6.set_ylabel(f'Average {target_col}')
        ax6.set_title('📈 Yearly Trend')
    else:
        # Daily trend
        daily_trend = df_temp.groupby(df_temp[datetime_col].dt.date)[target_col].mean()
        ax6.plot(daily_trend.index, daily_trend.values, alpha=0.7)
        ax6.set_xlabel('Date')
        ax6.set_ylabel(f'Average {target_col}')
        ax6.set_title('📈 Daily Trend')
        ax6.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(EDAConfig.OUTPUT_DIR / 'temporal_analysis.png', dpi=DPI, bbox_inches='tight')
    plt.show()
    
    # Print insights
    print(f"\n🔍 Key Temporal Insights:")
    
    peak_hour = hourly.idxmax()
    low_hour = hourly.idxmin()
    print(f"   • Peak hour: {peak_hour}:00 (avg: {hourly[peak_hour]:,.0f})")
    print(f"   • Lowest hour: {low_hour}:00 (avg: {hourly[low_hour]:,.0f})")
    
    peak_day = daily.idxmax()
    low_day = daily.idxmin()
    print(f"   • Peak day: {peak_day} (avg: {daily[peak_day]:,.0f})")
    print(f"   • Lowest day: {low_day} (avg: {daily[low_day]:,.0f})")
    
    weekday_avg = df_temp[~df_temp['is_weekend']][target_col].mean()
    weekend_avg = df_temp[df_temp['is_weekend']][target_col].mean()
    diff_pct = (weekday_avg - weekend_avg) / weekend_avg * 100
    print(f"   • Weekday avg: {weekday_avg:,.0f}")
    print(f"   • Weekend avg: {weekend_avg:,.0f}")
    print(f"   • Weekday is {abs(diff_pct):.1f}% {'higher' if diff_pct > 0 else 'lower'} than weekend")
    
    return {
        'peak_hour': int(peak_hour),
        'low_hour': int(low_hour),
        'peak_day': peak_day,
        'weekday_avg': weekday_avg,
        'weekend_avg': weekend_avg,
    }


# =============================================================================
# 5. DISTRIBUTION ANALYSIS
# =============================================================================

def distribution_analysis(df: pd.DataFrame, max_cols: int = 12) -> None:
    """
    Analyze distributions of all numeric columns.
    
    Args:
        df: Input DataFrame
        max_cols: Maximum columns to plot
    """
    print_section("5. DISTRIBUTION ANALYSIS")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numeric_cols:
        print("⚠️ No numeric columns found")
        return
    
    # Select columns to plot
    cols_to_plot = numeric_cols[:max_cols]
    n_cols = len(cols_to_plot)
    
    print(f"\n📊 Analyzing {n_cols} numeric columns")
    
    # Calculate grid dimensions
    n_rows = (n_cols + 3) // 4
    n_plot_cols = min(4, n_cols)
    
    fig, axes = plt.subplots(n_rows, n_plot_cols, figsize=(4*n_plot_cols, 3*n_rows))
    axes = axes.flatten() if n_cols > 1 else [axes]
    
    for i, col in enumerate(cols_to_plot):
        ax = axes[i]
        data = df[col].dropna()
        
        # Plot histogram with KDE
        sns.histplot(data, bins=30, kde=True, ax=ax, color=COLORS['primary'])
        
        # Add mean line
        ax.axvline(data.mean(), color='red', linestyle='--', alpha=0.7)
        
        # Title with skewness
        skew = data.skew()
        ax.set_title(f'{col}\n(skew: {skew:.2f})', fontsize=10)
        ax.set_xlabel('')
    
    # Hide empty subplots
    for i in range(n_cols, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(EDAConfig.OUTPUT_DIR / 'distributions.png', dpi=DPI, bbox_inches='tight')
    plt.show()
    
    # Print skewness summary
    print(f"\n📈 Skewness Summary:")
    for col in numeric_cols:
        skew = df[col].skew()
        if abs(skew) > 1:
            print(f"   • {col}: {skew:.2f} (highly skewed - consider transformation)")
        elif abs(skew) > 0.5:
            print(f"   • {col}: {skew:.2f} (moderately skewed)")


# =============================================================================
# 6. CORRELATION ANALYSIS
# =============================================================================

def correlation_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze correlations between features.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Correlation matrix
    """
    print_section("6. CORRELATION ANALYSIS")
    
    # Select numeric columns only
    numeric_df = df.select_dtypes(include=[np.number])
    
    if numeric_df.empty:
        print("⚠️ No numeric columns for correlation analysis")
        return pd.DataFrame()
    
    # Calculate correlation matrix
    corr_matrix = numeric_df.corr()
    
    print(f"\n📊 Analyzing correlations among {len(numeric_df.columns)} numeric features")
    
    # Visualizations
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)
    
    # Full heatmap
    ax1 = axes[0]
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    
    # Limit to 20 features for readability
    if len(corr_matrix) > 20:
        # Select most important features (highest variance or correlation with target)
        target_col = EDAConfig.TARGET_COL
        if target_col in corr_matrix.columns:
            top_features = corr_matrix[target_col].abs().sort_values(ascending=False).head(20).index
            corr_subset = corr_matrix.loc[top_features, top_features]
            mask = np.triu(np.ones_like(corr_subset, dtype=bool), k=1)
            sns.heatmap(corr_subset, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                       center=0, ax=ax1, square=True, annot_kws={'size': 8})
            ax1.set_title(f'Correlation Heatmap\n(Top 20 features by correlation with {target_col})')
        else:
            corr_subset = corr_matrix.iloc[:20, :20]
            mask = np.triu(np.ones_like(corr_subset, dtype=bool), k=1)
            sns.heatmap(corr_subset, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                       center=0, ax=ax1, square=True, annot_kws={'size': 8})
            ax1.set_title('Correlation Heatmap (First 20 features)')
    else:
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                   center=0, ax=ax1, square=True, annot_kws={'size': 8})
        ax1.set_title('Feature Correlation Heatmap')
    
    # Target correlations
    ax2 = axes[1]
    target_col = EDAConfig.TARGET_COL
    if target_col in corr_matrix.columns:
        target_corr = corr_matrix[target_col].drop(target_col).sort_values()
        colors = [COLORS['danger'] if x < 0 else COLORS['primary'] for x in target_corr.values]
        ax2.barh(range(len(target_corr)), target_corr.values, color=colors)
        ax2.set_yticks(range(len(target_corr)))
        ax2.set_yticklabels(target_corr.index, fontsize=8)
        ax2.set_xlabel('Correlation')
        ax2.set_title(f'Correlation with {target_col}')
        ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    else:
        ax2.text(0.5, 0.5, 'Target column not found', ha='center', va='center')
    
    plt.tight_layout()
    plt.savefig(EDAConfig.OUTPUT_DIR / 'correlations.png', dpi=DPI, bbox_inches='tight')
    plt.show()
    
    # Print strong correlations
    threshold = EDAConfig.CORRELATION_THRESHOLD
    print(f"\n🔗 Strong Correlations (|r| > {threshold}):")
    
    # Get upper triangle indices
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    strong_corr = []
    
    for col in upper_tri.columns:
        for idx in upper_tri.index:
            val = upper_tri.loc[idx, col]
            if pd.notna(val) and abs(val) > threshold:
                strong_corr.append((idx, col, val))
    
    strong_corr.sort(key=lambda x: abs(x[2]), reverse=True)
    
    for feat1, feat2, corr in strong_corr[:15]:
        direction = "positive" if corr > 0 else "negative"
        print(f"   • {feat1} ↔ {feat2}: {corr:.3f} ({direction})")
    
    if not strong_corr:
        print(f"   No correlations above threshold {threshold}")
    
    # Target correlations
    if target_col in corr_matrix.columns:
        print(f"\n🎯 Top Correlations with {target_col}:")
        target_corr = corr_matrix[target_col].drop(target_col).abs().sort_values(ascending=False)
        for feat, corr in target_corr.head(10).items():
            actual_corr = corr_matrix.loc[feat, target_col]
            print(f"   • {feat}: {actual_corr:.3f}")
    
    return corr_matrix


# =============================================================================
# 7. OUTLIER ANALYSIS
# =============================================================================

def outlier_analysis(df: pd.DataFrame) -> Dict:
    """
    Detect and analyze outliers.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary with outlier information
    """
    print_section("7. OUTLIER ANALYSIS")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numeric_cols:
        print("⚠️ No numeric columns for outlier analysis")
        return {}
    
    outlier_info = {}
    
    print(f"\n🔍 Analyzing outliers using IQR method:")
    
    for col in numeric_cols:
        data = df[col].dropna()
        
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = data[(data < lower_bound) | (data > upper_bound)]
        outlier_pct = len(outliers) / len(data) * 100
        
        outlier_info[col] = {
            'count': len(outliers),
            'percentage': outlier_pct,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
        }
        
        if outlier_pct > 1:  # Only show if more than 1%
            print(f"   • {col}: {len(outliers):,} outliers ({outlier_pct:.1f}%)")
    
    # Visualize outliers for key columns
    target_col = EDAConfig.TARGET_COL
    cols_to_plot = [target_col] if target_col in numeric_cols else numeric_cols[:4]
    
    fig, axes = plt.subplots(1, len(cols_to_plot), figsize=(4*len(cols_to_plot), 5))
    if len(cols_to_plot) == 1:
        axes = [axes]
    
    for ax, col in zip(axes, cols_to_plot):
        data = df[col].dropna()
        
        bp = ax.boxplot(data, patch_artist=True)
        bp['boxes'][0].set_facecolor(COLORS['primary'])
        
        # Highlight outlier count
        info = outlier_info[col]
        ax.set_title(f'{col}\n({info["count"]:,} outliers, {info["percentage"]:.1f}%)')
    
    plt.tight_layout()
    plt.savefig(EDAConfig.OUTPUT_DIR / 'outliers.png', dpi=DPI, bbox_inches='tight')
    plt.show()
    
    return outlier_info


# =============================================================================
# 8. FEATURE RELATIONSHIPS
# =============================================================================

def feature_relationships(df: pd.DataFrame) -> None:
    """
    Analyze relationships between features and target.
    
    Args:
        df: Input DataFrame
    """
    print_section("8. FEATURE RELATIONSHIPS")
    
    target_col = EDAConfig.TARGET_COL
    
    if target_col not in df.columns:
        print(f"⚠️ Target column '{target_col}' not found")
        return
    
    # Select numeric features
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != target_col]
    
    # Select top correlated features
    corr_with_target = df[numeric_cols].corrwith(df[target_col]).abs().sort_values(ascending=False)
    top_features = corr_with_target.head(6).index.tolist()
    
    if not top_features:
        print("⚠️ No suitable features for relationship analysis")
        return
    
    print(f"\n📊 Analyzing relationships for top {len(top_features)} correlated features")
    
    # Scatter plots
    n_cols = 3
    n_rows = 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    axes = axes.flatten()
    
    for i, col in enumerate(top_features):
        ax = axes[i]
        
        # Sample data if too large
        sample_size = min(5000, len(df))
        sample_df = df.sample(sample_size) if len(df) > sample_size else df
        
        ax.scatter(sample_df[col], sample_df[target_col], alpha=0.3, s=10)
        
        # Add trend line
        z = np.polyfit(sample_df[col].dropna(), sample_df[target_col].dropna(), 1)
        p = np.poly1d(z)
        x_line = np.linspace(sample_df[col].min(), sample_df[col].max(), 100)
        ax.plot(x_line, p(x_line), color='red', linestyle='--', linewidth=2)
        
        corr = df[col].corr(df[target_col])
        ax.set_title(f'{col}\n(r = {corr:.3f})')
        ax.set_xlabel(col)
        ax.set_ylabel(target_col)
    
    plt.tight_layout()
    plt.savefig(EDAConfig.OUTPUT_DIR / 'feature_relationships.png', dpi=DPI, bbox_inches='tight')
    plt.show()


# =============================================================================
# 9. CATEGORICAL ANALYSIS
# =============================================================================

def categorical_analysis(df: pd.DataFrame) -> None:
    """
    Analyze categorical variables.
    
    Args:
        df: Input DataFrame
    """
    print_section("9. CATEGORICAL VARIABLE ANALYSIS")
    
    # Identify categorical columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Also include low-cardinality numeric columns
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].nunique() <= 10 and col not in ['year', 'month', 'day', 'hour']:
            if col not in cat_cols:
                cat_cols.append(col)
    
    if not cat_cols:
        print("⚠️ No categorical columns found")
        return
    
    print(f"\n📊 Found {len(cat_cols)} categorical columns")
    
    target_col = EDAConfig.TARGET_COL
    
    for col in cat_cols[:6]:  # Limit to 6 columns
        print(f"\n📍 {col}:")
        value_counts = df[col].value_counts()
        print(f"   Unique values: {len(value_counts)}")
        print(f"   Top 5:")
        for val, count in value_counts.head(5).items():
            print(f"      • {val}: {count:,} ({count/len(df)*100:.1f}%)")
    
    # Visualize categorical vs target
    cols_to_plot = [c for c in cat_cols if df[c].nunique() <= 10][:4]
    
    if cols_to_plot and target_col in df.columns:
        fig, axes = plt.subplots(2, 2, figsize=FIGSIZE_LARGE)
        axes = axes.flatten()
        
        for i, col in enumerate(cols_to_plot):
            ax = axes[i]
            
            # Box plot
            order = df.groupby(col)[target_col].median().sort_values(ascending=False).index
            sns.boxplot(data=df, x=col, y=target_col, order=order, ax=ax)
            ax.set_title(f'{target_col} by {col}')
            ax.tick_params(axis='x', rotation=45)
        
        for i in range(len(cols_to_plot), 4):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(EDAConfig.OUTPUT_DIR / 'categorical_analysis.png', dpi=DPI, bbox_inches='tight')
        plt.show()


# =============================================================================
# 10. KEY INSIGHTS SUMMARY
# =============================================================================

def generate_insights(df: pd.DataFrame, results: Dict) -> List[str]:
    """
    Generate actionable insights from analysis.
    
    Args:
        df: Input DataFrame
        results: Dictionary with analysis results
        
    Returns:
        List of insight strings
    """
    print_section("10. KEY INSIGHTS & RECOMMENDATIONS")
    
    insights = []
    
    # Data quality
    missing_pct = df.isnull().sum().sum() / df.size * 100
    if missing_pct > 5:
        insights.append(f"⚠️ Data has {missing_pct:.1f}% missing values - consider imputation strategies")
    else:
        insights.append(f"✅ Data quality is good with only {missing_pct:.1f}% missing values")
    
    # Target variable
    target_col = EDAConfig.TARGET_COL
    if target_col in df.columns:
        skew = df[target_col].skew()
        if abs(skew) > 1:
            insights.append(f"📊 Target variable is highly skewed ({skew:.2f}) - consider log transformation")
    
    # Temporal patterns
    if 'temporal' in results:
        temporal = results['temporal']
        insights.append(f"🕐 Peak traffic hour: {temporal.get('peak_hour', 'N/A')}:00")
        insights.append(f"📅 Peak traffic day: {temporal.get('peak_day', 'N/A')}")
        
        weekday_avg = temporal.get('weekday_avg', 0)
        weekend_avg = temporal.get('weekend_avg', 0)
        if weekday_avg and weekend_avg:
            diff = abs(weekday_avg - weekend_avg) / weekend_avg * 100
            insights.append(f"📆 Weekday traffic is {diff:.0f}% {'higher' if weekday_avg > weekend_avg else 'lower'} than weekend")
    
    # Feature importance hints
    insights.append("🎯 Consider time-based features (hour, day, rush_hour) for model training")
    insights.append("🌤️ Weather features show correlation with traffic - include in model")
    
    # Print insights
    print("\n🔮 Actionable Insights:\n")
    for i, insight in enumerate(insights, 1):
        print(f"   {i}. {insight}")
    
    # Recommendations
    print("\n📋 Recommendations for Model Training:\n")
    recommendations = [
        "1. Use temporal features: hour, day_of_week, is_weekend, is_rush_hour",
        "2. Include rolling statistics for traffic_volume",
        "3. Consider one-hot encoding for weather categories",
        "4. Handle outliers with capping at 1.5*IQR",
        "5. Use TimeSeriesSplit for cross-validation",
        "6. Try ensemble methods (Random Forest, Gradient Boosting)",
    ]
    for rec in recommendations:
        print(f"   {rec}")
    
    return insights


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def run_eda(data_path: Optional[str] = None, save_outputs: bool = True) -> Dict:
    """
    Run complete EDA pipeline.
    
    Args:
        data_path: Optional path to data file
        save_outputs: Whether to save plots and reports
        
    Returns:
        Dictionary with all analysis results
    """
    print("=" * 60)
    print("🚦 SMART TRAFFIC MANAGEMENT SYSTEM")
    print("📊 EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create output directory
    if save_outputs:
        EDAConfig.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        print(f"📁 Outputs will be saved to: {EDAConfig.OUTPUT_DIR}")
    
    # Load data
    if data_path:
        df = load_data(Path(data_path))
    else:
        df = load_data()
    
    # Run all analyses
    results = {}
    
    results['overview'] = data_overview(df)
    results['missing'] = missing_value_analysis(df)
    results['target'] = target_analysis(df)
    results['temporal'] = temporal_analysis(df)
    distribution_analysis(df)
    results['correlation'] = correlation_analysis(df)
    results['outliers'] = outlier_analysis(df)
    feature_relationships(df)
    categorical_analysis(df)
    results['insights'] = generate_insights(df, results)
    
    print("\n" + "=" * 60)
    print("✅ EDA COMPLETE")
    print("=" * 60)
    print(f"⏰ Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if save_outputs:
        print(f"📁 Plots saved to: {EDAConfig.OUTPUT_DIR}")
    
    return results


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run EDA on traffic data")
    parser.add_argument('--data', '-d', type=str, help='Path to data file')
    parser.add_argument('--no-save', action='store_true', help='Do not save outputs')
    
    args = parser.parse_args()
    
    results = run_eda(
        data_path=args.data,
        save_outputs=not args.no_save
    )
