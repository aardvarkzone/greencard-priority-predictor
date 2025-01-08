from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
import pandas as pd
import numpy as np
from typing import Dict, Optional
import torch
from pathlib import Path
import json
import logging
from model import PriorityDatePredictor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Priority Date Predictor API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictionRequest(BaseModel):
    priority_date: str
    category: str
    current_date: Optional[str] = None

class PredictionResponse(BaseModel):
    estimated_wait_months: float
    confidence: float
    category: str
    priority_date: str
    estimated_current_date: str

# Load models
models: Dict[str, PriorityDatePredictor] = {}

def load_models():
    """Load all trained models."""
    model_dir = Path("data/models/latest/checkpoints")
    if not model_dir.exists():
        raise RuntimeError("Model directory not found")
    
    categories = [
        'eb1_india', 'eb2_india', 'eb3_india',
        'f1_india', 'f2a_india', 'f2b_india', 'f3_india', 'f4_india'
    ]
    
    for category in categories:
        model_path = model_dir / f"{category}_model.pt"
        if model_path.exists():
            try:
                models[category] = PriorityDatePredictor.load(str(model_path))
                logger.info(f"Loaded model for {category}")
            except Exception as e:
                logger.error(f"Error loading model for {category}: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """Load models on startup."""
    load_models()

def prepare_prediction_data(priority_date: str, current_date: str, category: str) -> pd.DataFrame:
    """Prepare data for prediction."""
    # Load historical data
    df = pd.read_csv(f'data/processed/processed_{"employment" if "eb" in category else "family"}.csv')
    df['date'] = pd.to_datetime(df['date'])
    
    # Convert dates
    pd_date = pd.to_datetime(priority_date)
    current = pd.to_datetime(current_date)
    
    # Calculate current movement
    movement_days = (current - pd_date).days
    
    # Create prediction row
    new_row = df.iloc[-1].copy()
    new_row['date'] = current
    new_row[f'{category}_movement'] = movement_days
    
    # Append to historical data
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    return df

@app.post("/predict", response_model=PredictionResponse)
async def predict_priority_date(request: PredictionRequest):
    """Make a prediction for priority date movement."""
    try:
        # Validate category
        if request.category not in models:
            raise HTTPException(status_code=400, detail=f"Invalid category: {request.category}")
        
        # Use current date if not provided
        current_date = request.current_date or datetime.now().strftime('%Y-%m-%d')
        
        # Prepare data
        df = prepare_prediction_data(
            request.priority_date,
            current_date,
            request.category
        )
        
        # Get prediction
        predictor = models[request.category]
        X = predictor.prepare_single_prediction(df)
        predicted_movement = predictor.predict_single(X)
        
        # Calculate results
        pd_date = pd.to_datetime(request.priority_date)
        current = pd.to_datetime(current_date)
        estimated_months = predicted_movement / 30.44  # Average days per month
        
        estimated_date = current + pd.Timedelta(days=predicted_movement)
        
        # Calculate confidence based on model metrics
        with open('data/models/latest/metrics/all_metrics.json', 'r') as f:
            metrics = json.load(f)
        
        rmse = metrics[request.category]['rmse']
        max_movement = df[f'{request.category}_movement'].abs().max()
        confidence = max(0, min(100, 100 * (1 - rmse / max_movement)))
        
        return PredictionResponse(
            estimated_wait_months=float(estimated_months),
            confidence=float(confidence),
            category=request.category,
            priority_date=request.priority_date,
            estimated_current_date=estimated_date.strftime('%Y-%m-%d')
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "models_loaded": list(models.keys())
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)