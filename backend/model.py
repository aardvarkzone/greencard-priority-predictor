import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import logging
from typing import Tuple, Dict, Optional, List

logger = logging.getLogger(__name__)

class EBVisaDataset(Dataset):
    """Simplified dataset for employment-based visa predictions."""
    
    def __init__(self, 
                 df: pd.DataFrame, 
                 category: str, 
                 sequence_length: int = 6,
                 scaler: Optional[StandardScaler] = None):
        df = df.copy()
        
        # Keep only essential features
        self.features = [
            f'{category}_movement',          # Base movement
            f'{category}_move_3m',          # 3-month trend
            f'{category}_vol_3m',           # 3-month volatility
            f'{category}_move_6m',          # 6-month trend
            f'{category}_vol_6m',           # 6-month volatility
            'fiscal_progress',              # Position in fiscal year
            'month_sin',                    # Cyclic month encoding
            'month_cos'
        ]
        
        # Add simple status encoding
        status_values = ['NORMAL', 'CURRENT', 'UNAVAILABLE']
        for status in status_values:
            col_name = f'{category}_status_{status}'
            df[col_name] = (df[f'{category}_status'] == status).astype(float)
            self.features.append(col_name)
        
        # Clean and prepare data
        self.data = df[self.features].copy()
        self.data = (self.data
                    .replace([np.inf, -np.inf], np.nan)
                    .ffill()
                    .bfill()
                    .fillna(0))
        
        # Scale features
        self.scaler = scaler if scaler is not None else StandardScaler()
        if scaler is None:
            self.data_scaled = self.scaler.fit_transform(self.data)
        else:
            self.data_scaled = self.scaler.transform(self.data)
        
        # Create sequences
        self.X, self.y = [], []
        movements = df[f'{category}_movement'].values
        
        for i in range(len(self.data_scaled) - sequence_length):
            self.X.append(self.data_scaled[i:(i + sequence_length)])
            next_movement = movements[i + sequence_length]
            self.y.append(np.clip(next_movement, -30, 30))
            
    def __len__(self) -> int:
        return len(self.X)
        
    def __getitem__(self, idx: int) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        return torch.FloatTensor(self.X[idx]), torch.FloatTensor([self.y[idx]])
    
    @property
    def input_size(self) -> int:
        return len(self.features)

class SimpleEBModel(nn.Module):
    """Simplified LSTM model for EB visa predictions."""
    
    def __init__(self, input_size: int, hidden_size: int = 64):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )
        
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, 1)
        
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        out = lstm_out[:, -1, :]  # Take last output
        
        out = F.relu(self.fc1(out))
        out = self.dropout(out)
        
        return self.fc2(out)

class EBVisaPredictor:
    """Simplified predictor for employment-based visas."""
    
    CONFIGS = {
        'eb1_india': {
            'sequence_length': 6,
            'hidden_size': 64,
            'learning_rate': 0.001,
            'batch_size': 32,
            'weight_decay': 0.01
        },
        'eb2_india': {
            'sequence_length': 6,
            'hidden_size': 64,
            'learning_rate': 0.001,
            'batch_size': 32,
            'weight_decay': 0.01
        },
        'eb3_india': {
            'sequence_length': 6,
            'hidden_size': 64,
            'learning_rate': 0.001,
            'batch_size': 32,
            'weight_decay': 0.01
        }
    }
    
    def __init__(self, category: str):
        self.category = category
        self.config = self.CONFIGS[category]
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.scaler = None
    
    def prepare_data(self, df: pd.DataFrame, val_size: float = 0.2) -> Tuple[DataLoader, DataLoader]:
        """Prepare dataloaders."""
        df = df.sort_values('date').reset_index(drop=True)
        train_size = int(len(df) * (1 - val_size))
        train_df = df.iloc[:train_size].copy()
        val_df = df.iloc[train_size:].copy()
        
        train_dataset = EBVisaDataset(
            train_df,
            self.category,
            self.config['sequence_length']
        )
        self.scaler = train_dataset.scaler
        
        val_dataset = EBVisaDataset(
            val_df,
            self.category,
            self.config['sequence_length'],
            self.scaler
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False
        )
        
        return train_loader, val_loader
    
    def train(self,
            train_loader: DataLoader,
            val_loader: DataLoader,
            epochs: int = 100,
            patience: int = 15) -> Dict[str, List[float]]:
        """Train the model."""
        input_size = train_loader.dataset.input_size
        self.model = SimpleEBModel(
            input_size=input_size,
            hidden_size=self.config['hidden_size']
        ).to(self.device)
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        best_val_loss = float('inf')
        best_model = None
        patience_counter = 0
        history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0
            
            for X, y in train_loader:
                X, y = X.to(self.device), y.to(self.device)
                
                optimizer.zero_grad()
                pred = self.model(X)
                loss = criterion(pred, y)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation phase
            self.model.eval()
            val_loss = 0
            val_predictions = []
            val_targets = []
            
            with torch.no_grad():
                for X, y in val_loader:
                    X, y = X.to(self.device), y.to(self.device)
                    pred = self.model(X)
                    loss = criterion(pred, y)
                    val_loss += loss.item()
                    
                    val_predictions.extend(pred.cpu().numpy())
                    val_targets.extend(y.cpu().numpy())
            
            val_loss /= len(val_loader)
            scheduler.step(val_loss)
            
            # Save history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = self.model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
            
            if (epoch + 1) % 5 == 0:
                val_predictions = np.array(val_predictions)
                val_targets = np.array(val_targets)
                within_5 = np.mean(np.abs(val_predictions - val_targets) <= 5) * 100
                within_10 = np.mean(np.abs(val_predictions - val_targets) <= 10) * 100
                
                logger.info(
                    f"Epoch {epoch+1}: "
                    f"Train Loss = {train_loss:.4f}, "
                    f"Val Loss = {val_loss:.4f}, "
                    f"Within 5 days: {within_5:.1f}%, "
                    f"Within 10 days: {within_10:.1f}%"
                )
        
        # Load best model
        if best_model is not None:
            self.model.load_state_dict(best_model)
        
        return history
    
    def save(self, path: str):
        """Save the model."""
        if self.model is None:
            raise ValueError("No model to save")
            
        save_dict = {
            'model_state': self.model.state_dict(),
            'scaler': self.scaler,
            'category': self.category,
            'config': self.config
        }
        torch.save(save_dict, path)
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'EBVisaPredictor':
        """Load a saved model."""
        save_dict = torch.load(path, map_location=torch.device('cpu'))
        
        predictor = cls(category=save_dict['category'])
        predictor.config = save_dict['config']
        
        input_size = len(save_dict['scaler'].get_params()['n_features_in_'])
        predictor.model = SimpleEBModel(
            input_size=input_size,
            hidden_size=predictor.config['hidden_size']
        )
        predictor.model.load_state_dict(save_dict['model_state'])
        predictor.model.to(predictor.device)
        
        predictor.scaler = save_dict['scaler']
        return predictor