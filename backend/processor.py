import pandas as pd
import numpy as np
from datetime import datetime
import os
import logging
from typing import List

class VisaBulletinProcessor:
    """Process visa bulletin data for time series analysis and prediction."""
    
    def __init__(self, log_level: int = logging.INFO):
        """Initialize the processor with logging configuration."""
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Constants
        self.MAX_FORWARD_MOVEMENT = 180   # Cap forward at 6 months
        self.MAX_BACKWARD_MOVEMENT = 365  # Cap backward at 1 year
        
        # Month mapping for date parsing
        self.MONTHS = {
            'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 
            'JUN': 6, 'JUNE': 6, 'JN': 6,
            'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12
        }
        
        # Common variations and special cases
        self.SPECIAL_VALUES = {
            'CURRENT': pd.Timestamp.max,
            'C': pd.Timestamp.max,
            'UNAVAILABLE': None,
            'UNAVAIL': None,
            'U': None,
            'UN': None,
            'N/A': None,
            '--': None
        }
        
    def parse_date(self, date_str: str) -> pd.Timestamp:
        """Parse priority dates from visa bulletin."""
        if not isinstance(date_str, str):
            return None
            
        # Clean the string
        date_str = date_str.upper()
        date_str = date_str.split('>')[0]  # Remove HTML remnants
        date_str = date_str.strip(' .,;:->')
        
        # Handle special values
        if date_str in self.SPECIAL_VALUES:
            return self.SPECIAL_VALUES[date_str]
            
        try:
            # Fix common typos
            if 'DEB' in date_str:
                date_str = date_str.replace('DEB', 'DEC')
            if 'O1' in date_str:  # Handle O vs 0
                date_str = date_str.replace('O1', '01')
                
            # Extract components - expect DDMMMYY format
            day = int(date_str[:2])
            month_str = date_str[2:5]
            year = int(date_str[5:])
            
            # Convert month string to number
            if month_str not in self.MONTHS:
                return None
            month = self.MONTHS[month_str]
            
            # Convert 2-digit year to full year
            if year < 50:  # Assume 20xx
                year += 2000
            elif year < 100:  # Assume 19xx
                year += 1900
                
            # Basic validation
            if not (1 <= day <= 31 and 1 <= month <= 12 and 1900 <= year <= 2030):
                return None
                
            return pd.Timestamp(year=year, month=month, day=day)
            
        except Exception as e:
            return None
            
    def add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features."""
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
        
        return df
        
    def process_category(self, df: pd.DataFrame, category: str) -> pd.DataFrame:
        """Process a single visa category."""
        df = df.copy()
        
        # Store original values
        df[f'{category}_original'] = df[category]
        
        # Convert dates to days from reference
        reference_date = pd.Timestamp('2025-01-01')  # Fixed reference point
        
        # Parse dates and log failures
        dates = []
        for idx, value in df[category].items():
            parsed = self.parse_date(value)
            dates.append(parsed)
            
        df[f'{category}_days'] = [(reference_date - d).days if d is not None else None for d in dates]
        
        # Calculate movements (negative since counting down)
        df[f'{category}_movement'] = -df[f'{category}_days'].diff()
        
        # Handle transitions to/from special statuses
        special_transition = (
            (df[f'{category}_days'] == 0) != (df[f'{category}_days'].shift() == 0) |  # Current transitions
            (df[f'{category}_days'].isna() != df[f'{category}_days'].shift().isna())   # Unavailable transitions
        )
        df.loc[special_transition, f'{category}_movement'] = 0
        
        # Cap extreme movements
        df[f'{category}_movement'] = df[f'{category}_movement'].clip(
            lower=-self.MAX_BACKWARD_MOVEMENT,
            upper=self.MAX_FORWARD_MOVEMENT
        )
        
        # Calculate rolling statistics
        for window in [3, 6, 12]:
            # Mean movement
            df[f'{category}_move_{window}m'] = (
                df[f'{category}_movement']
                .rolling(window, min_periods=1)
                .mean()
                .round(2)
            )
            
            # Movement volatility
            df[f'{category}_vol_{window}m'] = (
                df[f'{category}_movement']
                .rolling(window, min_periods=1)
                .std()
                .round(2)
            )
            
        # Identify retrogression (negative movement between valid dates)
        df[f'{category}_retrogression'] = (
            (df[f'{category}_movement'] < 0) & 
            (df[f'{category}_days'] != 0) & 
            (df[f'{category}_days'].shift() != 0) &
            (df[f'{category}_days'].notna()) & 
            (df[f'{category}_days'].shift().notna())
        )
        
        # Get statistics
        total_rows = len(df)
        valid_dates = df[f'{category}_days'].notna().sum()
        current_status = (df[f'{category}_days'] == 0).sum()
        unavailable = df[f'{category}_days'].isna().sum()
        retrogressions = df[f'{category}_retrogression'].sum()
        
        # Log statistics
        self.logger.info(f"\n{category} Statistics:")
        self.logger.info(f"Total rows: {total_rows}")
        self.logger.info(f"Valid dates: {valid_dates} ({valid_dates/total_rows*100:.1f}%)")
        self.logger.info(f"Current status: {current_status}")
        self.logger.info(f"Unavailable: {unavailable}")
        self.logger.info(f"Retrogression events: {retrogressions}")
        
        movements = df[df[f'{category}_movement'].notna()][f'{category}_movement']
        if len(movements) > 0:
            self.logger.info(f"Movement range: {movements.min():.0f} to {movements.max():.0f} days")
            
        return df
        
    def process_employment_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process employment-based visa data."""
        self.logger.info("\nProcessing employment-based data...")
        df = df.copy()
        
        # Sort by date
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        
        # Add temporal features
        df = self.add_temporal_features(df)
        
        # Process each category
        categories = ['eb1_india', 'eb2_india', 'eb3_india']
        for category in categories:
            df = self.process_category(df, category)
            
        return df
        
    def process_family_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process family-based visa data."""
        self.logger.info("\nProcessing family-based data...")
        df = df.copy()
        
        # Sort by date
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        
        # Add temporal features
        df = self.add_temporal_features(df)
        
        # Process each category
        categories = ['f1_india', 'f2a_india', 'f2b_india', 'f3_india', 'f4_india']
        for category in categories:
            df = self.process_category(df, category)
            
        return df

def main():
    """Process visa bulletin data."""
    try:
        processor = VisaBulletinProcessor()
        
        # Process employment data
        emp_df = pd.read_csv('data/raw/india_employment_based_movement.csv')
        processed_emp = processor.process_employment_data(emp_df)
        
        # Process family data
        fam_df = pd.read_csv('data/raw/india_family_based_movement.csv')
        processed_fam = processor.process_family_data(fam_df)
        
        # Save processed files
        os.makedirs('data/processed', exist_ok=True)
        processed_emp.to_csv('data/processed/processed_employment.csv', index=False)
        processed_fam.to_csv('data/processed/processed_family.csv', index=False)
        
        logging.info("\nProcessing completed successfully!")
        
    except Exception as e:
        logging.error(f"Error in processing: {str(e)}")
        raise

if __name__ == "__main__":
    main()