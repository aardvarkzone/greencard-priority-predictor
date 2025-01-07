import pandas as pd
import numpy as np
from datetime import datetime
import os
import logging
import sys
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

# Set up logging to output to console
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

@dataclass
class DataQualityMetrics:
    """Store data quality metrics for reporting."""
    total_rows: int
    valid_dates: int
    missing_dates: int
    retrogressions: int
    large_movements: int
    category: str
    date_range: Tuple[str, str]
    current_periods: int
    unavailable_periods: int

class VisaBulletinProcessor:
    """Process visa bulletin data for time series analysis and prediction."""
    
    def __init__(self):
        """Initialize the processor."""
        self.logger = logging.getLogger(__name__)
        
        # Constants
        self.MAX_FORWARD_MOVEMENT = 180   # Cap forward at 6 months
        self.MAX_BACKWARD_MOVEMENT = -365  # Cap backward at 1 year
        self.PROCESS_VERSION = "1.0.0"    # Version tracking
        
        # Month mapping for date parsing
        self.MONTHS = {
            'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 
            'JUN': 6, 'JUNE': 6, 'JN': 6,
            'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12
        }
        
        # Common variations and special cases
        self.SPECIAL_VALUES = {
            'CURRENT': 0,  # 0 days from reference = current
            'C': 0,
            'UNAVAILABLE': None,
            'UNAVAIL': None,
            'U': None,
            'UN': None,
            'N/A': None,
            '--': None
        }

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
        
        # Handle special values
        if date_str in self.SPECIAL_VALUES:
            if self.SPECIAL_VALUES[date_str] is None:
                return None
            elif self.SPECIAL_VALUES[date_str] == 0:  # CURRENT
                return pd.Timestamp('2025-01-01')  # Reference date
            return self.SPECIAL_VALUES[date_str]
            
        try:
            date = pd.to_datetime(date_str)
            if pd.isna(date):
                return None
            return date
        except:
            try:
                # Try parsing with format guideline
                return pd.to_datetime(date_str, format='%d%b%y', errors='coerce')
            except:
                return None

    def add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features."""
        print("Adding temporal features...")
        df = df.copy()
        
        # Ensure datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Basic time features
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        
        # Fiscal year (Oct-Sep)
        df['fiscal_year'] = df.apply(
            lambda x: x['year'] if x['month'] < 10 else x['year'] + 1, 
            axis=1
        )
        
        # Progress through fiscal year (0-1)
        fiscal_start = df.apply(
            lambda x: pd.Timestamp(
                year=x['year'] - 1 if x['month'] < 10 else x['year'],
                month=10, day=1
            ),
            axis=1
        )
        df['fiscal_progress'] = ((df['date'] - fiscal_start).dt.days / 365.25).round(4)
        
        # Add data quality period indicators
        df['data_quality'] = 'HIGH'
        df.loc[df['date'].dt.year < 2000, 'data_quality'] = 'LOW'
        df.loc[(df['date'].dt.year >= 2000) & 
               (df['date'].dt.year < 2004), 'data_quality'] = 'MEDIUM'
        
        return df

    def calculate_movement(self, df: pd.DataFrame, category: str) -> pd.Series:
        """Calculate real movements excluding status transitions."""
        movement = []
        prev_status = None
        prev_days = None
        
        for idx, row in df.iterrows():
            curr_status = row[f'{category}_status']
            curr_days = row[f'{category}_days']
            
            if prev_status == curr_status == 'NORMAL' and prev_days is not None:
                move = -(curr_days - prev_days)
                # Cap movement
                move = max(min(move, self.MAX_FORWARD_MOVEMENT), self.MAX_BACKWARD_MOVEMENT)
            else:
                move = 0  # No movement during status changes
                
            movement.append(move)
            prev_status = curr_status
            prev_days = curr_days
            
        return pd.Series(movement)

    def process_category(self, df: pd.DataFrame, category: str) -> Tuple[pd.DataFrame, DataQualityMetrics]:
        """Process a single visa category."""
        print(f"\nProcessing category: {category}")
        df = df.copy()
        
        # Store original values
        df[f'{category}_original'] = df[category]
        
        # Add status column
        df[f'{category}_status'] = 'NORMAL'
        df.loc[df[category].str.upper().isin(['CURRENT', 'C']), f'{category}_status'] = 'CURRENT'
        df.loc[df[category].str.upper().isin(['U', 'UN', 'UNAVAILABLE', 'UNAVAIL']), f'{category}_status'] = 'UNAVAILABLE'
        
        # Convert dates to days from reference
        reference_date = pd.Timestamp('2025-01-01')
        df[f'{category}_days'] = df[category].apply(lambda x: 
            (reference_date - self.parse_date(x)).days if self.parse_date(x) is not None else None
        )
        
        # Calculate movements excluding status transitions
        df[f'{category}_movement'] = self.calculate_movement(df, category)
        
        # Calculate rolling statistics for normal periods only
        normal_mask = df[f'{category}_status'] == 'NORMAL'
        for window in [3, 6, 12]:
            # Movement trends
            df[f'{category}_move_{window}m'] = (
                df.loc[normal_mask, f'{category}_movement']
                .rolling(window, min_periods=1)
                .mean()
                .round(2)
            )
            
            # Volatility
            df[f'{category}_vol_{window}m'] = (
                df.loc[normal_mask, f'{category}_movement']
                .rolling(window, min_periods=1)
                .std()
                .round(2)
            )
        
        # Identify real retrogressions (excluding status changes)
        df[f'{category}_retrogression'] = (
            (df[f'{category}_movement'] < 0) & 
            (df[f'{category}_status'] == 'NORMAL')
        )
        
        # Calculate metrics
        metrics = DataQualityMetrics(
            total_rows=len(df),
            valid_dates=df[f'{category}_days'].notna().sum(),
            missing_dates=df[f'{category}_days'].isna().sum(),
            retrogressions=df[f'{category}_retrogression'].sum(),
            large_movements=abs(df[f'{category}_movement']).gt(self.MAX_FORWARD_MOVEMENT/2).sum(),
            category=category,
            date_range=(
                df['date'].min().strftime('%Y-%m-%d'),
                df['date'].max().strftime('%Y-%m-%d')
            ),
            current_periods=(df[f'{category}_status'] == 'CURRENT').sum(),
            unavailable_periods=(df[f'{category}_status'] == 'UNAVAILABLE').sum()
        )
        
        print(f"Processed {metrics.total_rows} rows for {category}")
        print(f"Status breakdown: NORMAL={normal_mask.sum()}, "
              f"CURRENT={metrics.current_periods}, "
              f"UNAVAILABLE={metrics.unavailable_periods}")
        
        return df, metrics

    def process_employment_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, DataQualityMetrics]]:
        """Process employment-based visa data."""
        print("\nProcessing employment-based data...")
        
        # Validate input data
        self.validate_raw_data(df, ['date', 'eb1_india', 'eb2_india', 'eb3_india'])
        
        # Sort by date
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        
        # Add temporal features
        df = self.add_temporal_features(df)
        
        # Process each category
        quality_metrics = {}
        categories = ['eb1_india', 'eb2_india', 'eb3_india']
        for category in categories:
            df, metrics = self.process_category(df, category)
            quality_metrics[category] = metrics
        
        return df, quality_metrics

    def process_family_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, DataQualityMetrics]]:
        """Process family-based visa data."""
        print("\nProcessing family-based data...")
        
        # Validate input data
        self.validate_raw_data(df, ['date', 'f1_india', 'f2a_india', 'f2b_india', 'f3_india', 'f4_india'])
        
        # Sort by date
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        
        # Add temporal features
        df = self.add_temporal_features(df)
        
        # Process each category
        quality_metrics = {}
        categories = ['f1_india', 'f2a_india', 'f2b_india', 'f3_india', 'f4_india']
        for category in categories:
            df, metrics = self.process_category(df, category)
            quality_metrics[category] = metrics
        
        return df, quality_metrics

def main():
    """Process visa bulletin data."""
    print("\n=== Starting Visa Bulletin Data Processing ===\n")
    
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
        
        print(f"\nSaved processed files:")
        print(f"- {emp_output}")
        print(f"- {fam_output}")
        
        # Save metrics
        metrics_df = pd.DataFrame([
            {**vars(metric), 'data_type': 'employment' if 'eb' in metric.category else 'family'}
            for metrics in [emp_metrics, fam_metrics]
            for metric in metrics.values()
        ])
        
        metrics_output = os.path.join(processed_dir, 'quality_metrics.csv')
        metrics_df.to_csv(metrics_output, index=False)
        print(f"- {metrics_output}")
        
        print("\n=== Processing completed successfully! ===")
        
    except Exception as e:
        print(f"\nError during processing: {str(e)}")
        raise

if __name__ == "__main__":
    main()