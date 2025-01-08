import pandas as pd
import numpy as np
from datetime import datetime
import os
import logging
import sys
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from scipy import stats

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

@dataclass
class DataQualityMetrics:
    """Enhanced data quality metrics for reporting."""
    total_rows: int
    valid_dates: int
    missing_dates: int
    retrogressions: int
    large_movements: int
    category: str
    date_range: Tuple[str, str]
    current_periods: int
    unavailable_periods: int
    current_runs: int
    max_current_run: int
    avg_movement: float
    movement_volatility: float
    status_stability: float
    seasonal_pattern_strength: float

class VisaBulletinProcessor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Enhanced constants
        self.MAX_FORWARD_MOVEMENT = 180  # 6 months
        self.MAX_BACKWARD_MOVEMENT = -365  # 1 year
        self.TYPICAL_MOVEMENT = 30  # Typical monthly movement
        self.MIN_MOVEMENT = -30  # Minimum typical movement
        self.PROCESS_VERSION = "2.0.0"
        
        # Status values with enhanced mapping
        self.STATUS_VALUES = {
            'CURRENT': 'CURRENT',
            'C': 'CURRENT',
            'UNAVAILABLE': 'UNAVAILABLE',
            'UNAVAIL': 'UNAVAILABLE',
            'U': 'UNAVAILABLE',
            'UN': 'UNAVAILABLE',
            'N/A': 'UNAVAILABLE',
            '--': 'UNAVAILABLE'
        }
        
        # Enhanced month mapping
        self.MONTHS = {
            'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 
            'JUN': 6, 'JUNE': 6, 'JN': 6,
            'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12
        }

    def add_enhanced_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add enhanced time-based features."""
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        
        # Basic time features
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        df['day_of_year'] = df['date'].dt.dayofyear
        
        # Fiscal year features (Oct-Sep)
        df['fiscal_year'] = df.apply(
            lambda x: x['year'] if x['month'] < 10 else x['year'] + 1, 
            axis=1
        )
        
        # Fiscal year progress (0-1)
        fiscal_start = df.apply(
            lambda x: pd.Timestamp(
                year=x['year'] - 1 if x['month'] < 10 else x['year'],
                month=10, day=1
            ),
            axis=1
        )
        df['fiscal_progress'] = ((df['date'] - fiscal_start).dt.days / 365.25).round(4)
        
        # Cyclic features for better seasonal pattern capture
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['fiscal_month_sin'] = np.sin(2 * np.pi * df['fiscal_progress'])
        df['fiscal_month_cos'] = np.cos(2 * np.pi * df['fiscal_progress'])
        
        # Quarter end indicators
        df['is_quarter_end'] = df['month'].isin([3, 6, 9, 12]).astype(int)
        df['is_fiscal_quarter_end'] = df['month'].isin([12, 3, 6, 9]).astype(int)
        
        return df

    def calculate_enhanced_movement(self, df: pd.DataFrame, category: str) -> pd.DataFrame:
        """Calculate enhanced movement metrics."""
        movements = []
        status_changes = []
        days_since_change = []
        trend_strength = []
        current_runs = []
        
        prev_status = None
        prev_days = None
        change_count = 0
        current_streak = 0
        movement_window = []
        
        for idx, row in df.iterrows():
            curr_status = row[f'{category}_status']
            curr_days = row[f'{category}_days']
            
            # Status change tracking
            is_change = curr_status != prev_status if prev_status else False
            status_changes.append(is_change)
            
            if is_change:
                change_count = 0
            change_count += 1
            days_since_change.append(change_count)
            
            # Movement calculation with enhanced logic
            if prev_status == curr_status == 'NORMAL' and prev_days is not None and pd.notnull(curr_days):
                move = -(curr_days - prev_days)
                # Apply movement constraints
                move = max(min(move, self.MAX_FORWARD_MOVEMENT), self.MAX_BACKWARD_MOVEMENT)
                
                # Calculate trend strength
                movement_window.append(move)
                if len(movement_window) > 6:
                    movement_window.pop(0)
                    trend = stats.linregress(range(len(movement_window)), movement_window)
                    trend_strength.append(trend.rvalue)
                else:
                    trend_strength.append(0)
                    
            elif prev_status != 'CURRENT' and curr_status == 'CURRENT':
                move = self.MAX_FORWARD_MOVEMENT
                trend_strength.append(1)  # Strong positive trend
            elif prev_status == 'CURRENT' and curr_status != 'CURRENT':
                move = self.MAX_BACKWARD_MOVEMENT
                trend_strength.append(-1)  # Strong negative trend
            else:
                move = 0
                trend_strength.append(0)
            
            movements.append(move)
            
            # Track CURRENT runs
            if curr_status == 'CURRENT':
                current_streak += 1
            else:
                current_streak = 0
            current_runs.append(current_streak)
            
            # Update previous values
            prev_status = curr_status
            prev_days = curr_days
        
        # Calculate additional metrics
        df_temp = pd.DataFrame({
            'movement': movements,
            'status_changed': status_changes,
            'days_since_change': days_since_change,
            'trend_strength': trend_strength,
            'current_run_length': current_runs
        })
        
        # Add rolling statistics
        for window in [3, 6, 12]:
            df_temp[f'move_{window}m'] = (
                df_temp['movement'].rolling(window, min_periods=1).mean()
            )
            df_temp[f'vol_{window}m'] = (
                df_temp['movement'].rolling(window, min_periods=1).std()
            )
            df_temp[f'trend_{window}m'] = (
                df_temp['trend_strength'].rolling(window, min_periods=1).mean()
            )
        
        return df_temp

    def calculate_seasonal_patterns(self, df: pd.DataFrame, category: str) -> pd.DataFrame:
        """Calculate seasonal patterns in movement."""
        df = df.copy()
        
        # Group by month and calculate statistics
        monthly_stats = df.groupby('month')[f'{category}_movement'].agg([
            'mean', 'std', 'count'
        ]).reset_index()
        
        # Calculate seasonal indices
        overall_mean = df[f'{category}_movement'].mean()
        monthly_stats['seasonal_index'] = monthly_stats['mean'] / overall_mean
        
        # Map back to original dataframe
        month_to_index = dict(zip(monthly_stats['month'], monthly_stats['seasonal_index']))
        df[f'{category}_seasonal_index'] = df['month'].map(month_to_index)
        
        return df

    def add_status_transition_features(self, df: pd.DataFrame, category: str) -> pd.DataFrame:
        """Add features related to status transitions."""
        df = df.copy()
        
        # Calculate transition probabilities
        status_counts = df[f'{category}_status'].value_counts()
        transition_matrix = pd.crosstab(
            df[f'{category}_status'], 
            df[f'{category}_status'].shift(-1), 
            normalize='index'
        )
        
        # Add probability of next status
        for status in ['NORMAL', 'CURRENT', 'UNAVAILABLE']:
            if status in transition_matrix.columns:
                df[f'{category}_prob_{status}'] = df[f'{category}_status'].map(
                    lambda x: transition_matrix.loc[x, status] if x in transition_matrix.index else 0
                )
        
        return df

    def process_category(self, df: pd.DataFrame, category: str) -> Tuple[pd.DataFrame, DataQualityMetrics]:
        """Enhanced processing for a single visa category."""
        print(f"\nProcessing category: {category}")
        df = df.copy()
        
        # Basic processing
        df[f'{category}_original'] = df[category]
        df[f'{category}_status'] = df[category].str.upper().map(
            lambda x: self.STATUS_VALUES.get(x, 'NORMAL')
        )
        
        # Enhanced date parsing and processing
        df[f'{category}_parsed_date'] = df[category].apply(self.parse_date)
        reference_date = pd.Timestamp('2025-01-01')
        df[f'{category}_days'] = df[f'{category}_parsed_date'].apply(
            lambda x: (reference_date - x).days if pd.notnull(x) else None
        )
        
        # Calculate enhanced movements
        movement_data = self.calculate_enhanced_movement(df, category)
        for col in movement_data.columns:
            df[f'{category}_{col}'] = movement_data[col]
        
        # Add seasonal patterns
        df = self.calculate_seasonal_patterns(df, category)
        
        # Add status transition features
        df = self.add_status_transition_features(df, category)
        
        # Calculate enhanced metrics
        movement_series = df[f'{category}_movement']
        valid_movements = movement_series[movement_series != 0]
        
        metrics = DataQualityMetrics(
            total_rows=len(df),
            valid_dates=df[f'{category}_parsed_date'].notna().sum(),
            missing_dates=df[f'{category}_parsed_date'].isna().sum(),
            retrogressions=sum((df[f'{category}_movement'] < self.MIN_MOVEMENT) & 
                             (df[f'{category}_status'] == 'NORMAL')),
            large_movements=sum(abs(df[f'{category}_movement']) > self.TYPICAL_MOVEMENT),
            category=category,
            date_range=(
                df['date'].min().strftime('%Y-%m-%d'),
                df['date'].max().strftime('%Y-%m-%d')
            ),
            current_periods=(df[f'{category}_status'] == 'CURRENT').sum(),
            unavailable_periods=(df[f'{category}_status'] == 'UNAVAILABLE').sum(),
            current_runs=df[f'{category}_current_run_length'].astype(bool).sum(),
            max_current_run=df[f'{category}_current_run_length'].max(),
            avg_movement=valid_movements.mean(),
            movement_volatility=valid_movements.std(),
            status_stability=1 - (df[f'{category}_status_changed'].mean()),
            seasonal_pattern_strength=df[f'{category}_seasonal_index'].std()
        )
        
        return df, metrics

    def process_employment_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, DataQualityMetrics]]:
        """Process employment-based visa data with enhanced features."""
        print("\nProcessing employment-based data...")
        
        # Validate data
        self.validate_raw_data(df, ['date', 'eb1_india', 'eb2_india', 'eb3_india'])
        
        # Sort and add enhanced temporal features
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        df = self.add_enhanced_temporal_features(df)
        
        # Process categories
        quality_metrics = {}
        categories = ['eb1_india', 'eb2_india', 'eb3_india']
        
        for category in categories:
            df, metrics = self.process_category(df, category)
            quality_metrics[category] = metrics
        
        # Add enhanced cross-category features
        for i in range(len(categories)):
            for j in range(i+1, len(categories)):
                cat1, cat2 = categories[i], categories[j]
                
                # Movement correlations at different windows
                for window in [3, 6, 12]:
                    df[f'{cat1}_{cat2}_corr_{window}m'] = (
                        df[f'{cat1}_movement'].rolling(window).corr(df[f'{cat2}_movement'])
                    )
                
                # Status alignment
                df[f'{cat1}_{cat2}_same_status'] = (
                    df[f'{cat1}_status'] == df[f'{cat2}_status']
                ).astype(int)
                
                # Trend alignment
                df[f'{cat1}_{cat2}_trend_align'] = np.sign(
                    df[f'{cat1}_trend_strength'] * df[f'{cat2}_trend_strength']
                )
        
        return df, quality_metrics

    def process_family_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, DataQualityMetrics]]:
        """Process family-based visa data with enhanced features."""
        print("\nProcessing family-based data...")
        
        # Validate data
        self.validate_raw_data(df, ['date', 'f1_india', 'f2a_india', 'f2b_india', 'f3_india', 'f4_india'])
        
        # Sort and add enhanced temporal features
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        df = self.add_enhanced_temporal_features(df)
        
        # Process categories
        quality_metrics = {}
        categories = ['f1_india', 'f2a_india', 'f2b_india', 'f3_india', 'f4_india']
        
        for category in categories:
            df, metrics = self.process_category(df, category)
            quality_metrics[category] = metrics
        
        # Add enhanced cross-category features (same as employment)
        for i in range(len(categories)):
            for j in range(i+1, len(categories)):
                cat1, cat2 = categories[i], categories[j]
                
                # Movement correlations at different windows
                for window in [3, 6, 12]:
                    df[f'{cat1}_{cat2}_corr_{window}m'] = (
                        df[f'{cat1}_movement'].rolling(window).corr(df[f'{cat2}_movement'])
                    )
                
                # Status alignment
                df[f'{cat1}_{cat2}_same_status'] = (
                    df[f'{cat1}_status'] == df[f'{cat2}_status']
                ).astype(int)
                
                # Trend alignment
                df[f'{cat1}_{cat2}_trend_align'] = np.sign(
                    df[f'{cat1}_trend_strength'] * df[f'{cat2}_trend_strength']
                )
        
        return df, quality_metrics

    def validate_raw_data(self, df: pd.DataFrame, expected_columns: List[str]) -> None:
        """Validate raw input data structure and basic quality."""
        print(f"Validating data with {len(df)} rows...")
        
        # Check required columns
        missing_cols = set(expected_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        # Check for empty DataFrame
        if df.empty:
            raise ValueError("Input DataFrame is empty")
            
        # Check date column
        if not pd.to_datetime(df['date'], errors='coerce').notna().all():
            raise ValueError("Invalid dates found in date column")
            
        print("Raw data validation passed")

    def parse_date(self, date_str: str) -> Optional[pd.Timestamp]:
        """Parse priority dates from visa bulletin."""
        if not isinstance(date_str, str):
            return None
            
        date_str = date_str.upper().strip()
        
        # Check if it's a special status first
        if date_str in self.STATUS_VALUES:
            return None
            
        try:
            # Try standard date parsing
            date = pd.to_datetime(date_str)
            if pd.isna(date):
                return None
            return date
        except:
            try:
                # Try parsing priority date format
                return pd.to_datetime(date_str, format='%d%b%y', errors='coerce')
            except:
                return None

def main():
    """Process visa bulletin data with enhanced features."""
    print("\n=== Starting Enhanced Visa Bulletin Data Processing ===\n")
    
    try:
        # Get script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        print(f"Working directory: {script_dir}")
        
        # Define paths
        raw_dir = os.path.join(script_dir, 'data', 'raw')
        processed_dir = os.path.join(script_dir, 'data', 'processed')
        
        # Create output directory
        os.makedirs(processed_dir, exist_ok=True)
        print(f"Created/verified processed directory: {processed_dir}")
        
        # Initialize processor
        processor = VisaBulletinProcessor()
        
        # Process employment data
        emp_path = os.path.join(raw_dir, 'india_employment_based_movement.csv')
        print(f"\nReading employment data from: {emp_path}")
        emp_df = pd.read_csv(emp_path)
        processed_emp, emp_metrics = processor.process_employment_data(emp_df)
        
        # Process family data
        fam_path = os.path.join(raw_dir, 'india_family_based_movement.csv')
        print(f"\nReading family data from: {fam_path}")
        fam_df = pd.read_csv(fam_path)
        processed_fam, fam_metrics = processor.process_family_data(fam_df)
        
        # Save processed files
        emp_output = os.path.join(processed_dir, 'processed_employment.csv')
        fam_output = os.path.join(processed_dir, 'processed_family.csv')
        
        processed_emp.to_csv(emp_output, index=False)
        processed_fam.to_csv(fam_output, index=False)
        
        print(f"\nSaved enhanced processed files:")
        print(f"- {emp_output}")
        print(f"- {fam_output}")
        
        # Save enhanced metrics
        metrics_df = pd.DataFrame([
            {
                **vars(metric),
                'data_type': 'employment' if 'eb' in metric.category else 'family'
            }
            for metrics in [emp_metrics, fam_metrics]
            for metric in metrics.values()
        ])
        
        metrics_output = os.path.join(processed_dir, 'quality_metrics.csv')
        metrics_df.to_csv(metrics_output, index=False)
        print(f"- {metrics_output}")
        
        # Print enhanced summary statistics
        print("\n=== Enhanced Processing Summary ===")
        print("\nEmployment Categories:")
        for category, metrics in emp_metrics.items():
            print(f"\n{category}:")
            print(f"  Total rows: {metrics.total_rows}")
            print(f"  Valid dates: {metrics.valid_dates}")
            print(f"  CURRENT periods: {metrics.current_periods}")
            print(f"  Average movement: {metrics.avg_movement:.1f} days")
            print(f"  Movement volatility: {metrics.movement_volatility:.1f}")
            print(f"  Status stability: {metrics.status_stability:.2f}")
            print(f"  Seasonal strength: {metrics.seasonal_pattern_strength:.2f}")
            
        print("\nFamily Categories:")
        for category, metrics in fam_metrics.items():
            print(f"\n{category}:")
            print(f"  Total rows: {metrics.total_rows}")
            print(f"  Valid dates: {metrics.valid_dates}")
            print(f"  CURRENT periods: {metrics.current_periods}")
            print(f"  Average movement: {metrics.avg_movement:.1f} days")
            print(f"  Movement volatility: {metrics.movement_volatility:.1f}")
            print(f"  Status stability: {metrics.status_stability:.2f}")
            print(f"  Seasonal strength: {metrics.seasonal_pattern_strength:.2f}")
        
        print("\n=== Enhanced Processing completed successfully! ===")
        
    except Exception as e:
        print(f"\nError during processing: {str(e)}")
        raise

if __name__ == "__main__":
    main()