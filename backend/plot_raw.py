import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

def get_continuous_periods(dates, mask):
    """Find continuous periods where mask is True."""
    periods = []
    start_idx = None
    
    for i, (date, value) in enumerate(zip(dates, mask)):
        if value and start_idx is None:
            start_idx = i
        elif not value and start_idx is not None:
            periods.append((dates.iloc[start_idx], dates.iloc[i-1]))
            start_idx = None
            
    # Handle if we're still in a period at the end
    if start_idx is not None:
        periods.append((dates.iloc[start_idx], dates.iloc[-1]))
        
    return periods

def plot_category_data(df, categories, title, output_path):
    """Plot visa bulletin data with status indicators."""
    plt.figure(figsize=(15, 7))
    
    # Plot each category
    for category in categories:
        # Plot normal movement
        df[f'{category}_status'] = df[f'{category}_status'].fillna('NORMAL')
        normal_mask = df[f'{category}_status'] == 'NORMAL'
        if normal_mask.any():
            plt.plot(df.loc[normal_mask, 'date'], 
                    df.loc[normal_mask, f'{category}_days'],
                    label=category.split('_')[0].upper(),
                    linewidth=2)
        
        # Mark CURRENT periods
        current_mask = df[f'{category}_status'] == 'CURRENT'
        if current_mask.any():
            plt.plot(df.loc[current_mask, 'date'],
                    [0] * current_mask.sum(),
                    '--',
                    alpha=0.5,
                    linewidth=1)
        
        # Mark UNAVAILABLE periods
        unavail_mask = df[f'{category}_status'] == 'UNAVAILABLE'
        if unavail_mask.any():
            plt.plot(df.loc[unavail_mask, 'date'],
                    [df[f'{category}_days'].min()] * unavail_mask.sum(),
                    'rx',
                    alpha=0.5,
                    markersize=5)
    
    # Add data quality background colors
    quality_colors = {
        'LOW': '#ffcccc',      # Light red
        'MEDIUM': '#fff2cc',   # Light yellow
        'HIGH': '#ccffcc'      # Light green
    }
    
    for quality, color in quality_colors.items():
        mask = df['data_quality'] == quality
        if mask.any():
            plt.axvspan(df.loc[mask, 'date'].min(), 
                       df.loc[mask, 'date'].max(),
                       color=color,
                       alpha=0.2,
                       label=f'{quality} Quality Data')
    
    # Customize plot
    plt.title(title, fontsize=14, pad=20)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Days from Reference', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Create a single legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), 
              loc='upper right', fontsize=10)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Create enhanced visualizations of visa bulletin data."""
    try:
        # Read processed data
        print("Reading processed data...")
        emp_df = pd.read_csv('data/processed/processed_employment.csv')
        fam_df = pd.read_csv('data/processed/processed_family.csv')

        # Convert date columns to datetime
        print("Converting dates...")
        emp_df['date'] = pd.to_datetime(emp_df['date'])
        fam_df['date'] = pd.to_datetime(fam_df['date'])

        # Plot employment-based categories
        print("Plotting employment-based data...")
        plot_category_data(
            emp_df,
            ['eb1_india', 'eb2_india', 'eb3_india'],
            'Employment-Based Priority Dates Movement (India)',
            'data/processed/employment_trends.png'
        )

        # Plot family-based categories
        print("Plotting family-based data...")
        plot_category_data(
            fam_df,
            ['f1_india', 'f2a_india', 'f2b_india', 'f3_india', 'f4_india'],
            'Family-Based Priority Dates Movement (India)',
            'data/processed/family_trends.png'
        )
        
        print("Plotting completed successfully!")
        
    except Exception as e:
        print(f"Error during plotting: {str(e)}")
        raise

if __name__ == "__main__":
    main()