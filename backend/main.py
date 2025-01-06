from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
from typing import Optional
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(title="Green Card Priority Date Predictor API")

# Enable CORS with environment variables
app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("FRONTEND_URL", "http://localhost:3000")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictionRequest(BaseModel):
    priority_date: str
    visa_category: str
    country: str

class PredictionResponse(BaseModel):
    wait_time: int
    confidence: float
    input_data: PredictionRequest

@app.post("/predict", response_model=PredictionResponse)
async def predict_priority_date(request: PredictionRequest):
    try:
        # Validate date format
        datetime.strptime(request.priority_date, '%Y-%m-%d')
        
        # TODO: Add your ML model prediction logic here
        # This is just a placeholder that returns mock data
        wait_time = 18  # months
        confidence = 85.5  # percentage
        
        return PredictionResponse(
            wait_time=wait_time,
            confidence=confidence,
            input_data=request
        )
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 8000))
    )