import pandas as pd
import numpy as np
import torch
import logging
import os
from pathlib import Path
from datetime import datetime
import json
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, List

try:
    from model import EBVisaPredictor
except ImportError:
    from .model import EBVisaPredictor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)

def setup_training_environment() -> Path:
    """Setup directories for model artifacts."""
    base_dir = Path('data/models')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = base_dir / timestamp
    
    # Create subdirectories
    for subdir in ['checkpoints', 'plots', 'metrics']:
        (run_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Created training directory: {run_dir}")
    return run_dir

def plot_training_history(history: Dict[str, list], 
                         metrics: Dict[str, float],
                         category: str, 
                         save_path: Path):
    """Simple training visualization."""
    plt.figure(figsize=(10, 5))
    
    plt.plot(history['train_loss'], label='Training Loss', color='blue')
    plt.plot(history['val_loss'], label='Validation Loss', color='red')
    
    plt.title(f'Training History - {category}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add metrics annotations
    metrics_text = (
        f"RMSE: {metrics['rmse']:.2f}\n"
        f"Within 5 days: {metrics['within_5_days']:.1f}%\n"
        f"Within 10 days: {metrics['within_10_days']:.1f}%"
    )
    plt.text(0.02, 0.98, metrics_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plot_path = save_path / 'plots' / f'{category}_training.png'
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"Training plot saved to {plot_path}")

def evaluate_model(predictor: EBVisaPredictor, 
                  val_loader: torch.utils.data.DataLoader) -> Dict[str, float]:
    """Simple model evaluation."""
    predictor.model.eval()
    predictions = []
    targets = []
    
    with torch.no_grad():
        for X, y in val_loader:
            X = X.to(predictor.device)
            y = y.to(predictor.device)
            pred = predictor.model(X)
            
            predictions.extend(pred.cpu().numpy())
            targets.extend(y.cpu().numpy())
    
    predictions = np.array(predictions)
    targets = np.array(targets)
    errors = targets - predictions
    abs_errors = np.abs(errors)
    
    metrics = {
        'rmse': float(np.sqrt(np.mean(errors ** 2))),
        'mae': float(np.mean(abs_errors)),
        'within_5_days': float(np.mean(abs_errors <= 5) * 100),
        'within_10_days': float(np.mean(abs_errors <= 10) * 100)
    }
    
    return metrics

def train_category(df: pd.DataFrame, 
                  category: str,
                  run_dir: Path) -> Dict[str, float]:
    """Train model for an EB visa category."""
    logger.info(f"\nTraining model for {category}")
    
    predictor = EBVisaPredictor(category=category)
    
    # Prepare data
    train_loader, val_loader = predictor.prepare_data(df, val_size=0.2)
    
    # Train model
    history = predictor.train(train_loader, val_loader, epochs=100, patience=15)
    
    # Evaluate model
    metrics = evaluate_model(predictor, val_loader)
    
    # Save artifacts
    plot_training_history(history, metrics, category, run_dir)
    predictor.save(run_dir / 'checkpoints' / f'{category}_model.pt')
    
    with open(run_dir / 'metrics' / f'{category}_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Log metrics
    logger.info(f"Model metrics for {category}:")
    for metric, value in metrics.items():
        logger.info(f"{metric}: {value:.4f}")
    
    return metrics

def main():
    """Main training pipeline."""
    try:
        # Set random seeds
        torch.manual_seed(42)
        np.random.seed(42)
        torch.backends.cudnn.deterministic = True
        
        # Setup environment
        run_dir = setup_training_environment()
        logger.info(f"Models will be saved to: {run_dir}")
        
        # Load dataset
        df = pd.read_csv('data/processed/processed_employment.csv')
        df['date'] = pd.to_datetime(df['date'])
        
        # Train models
        categories = ['eb1_india', 'eb2_india', 'eb3_india']
        all_metrics = {}
        
        for category in categories:
            try:
                metrics = train_category(df, category, run_dir)
                all_metrics[category] = metrics
            except Exception as e:
                logger.error(f"Error training {category}: {str(e)}")
                logger.exception("Detailed traceback:")
                continue
        
        # Save overall metrics
        metrics_path = run_dir / 'metrics' / 'all_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(all_metrics, f, indent=4)
        
        # Plot comparison
        plt.figure(figsize=(10, 5))
        categories = list(all_metrics.keys())
        metrics = ['rmse', 'within_10_days']
        metric_names = {'rmse': 'RMSE (days)', 'within_10_days': 'Within 10 days (%)'}
        
        for i, metric in enumerate(metrics):
            plt.subplot(1, 2, i+1)
            values = [m[metric] for m in all_metrics.values()]
            plt.bar(categories, values)
            plt.title(metric_names[metric])
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(run_dir / 'plots' / 'category_comparison.png')
        plt.close()
        
        logger.info("\nTraining completed successfully!")
        logger.info(f"All model artifacts saved to: {run_dir}")
        
    except Exception as e:
        logger.error(f"Critical error in training pipeline: {str(e)}")
        logger.exception("Detailed traceback:")
        raise

if __name__ == "__main__":
    main()