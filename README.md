# Greencard Priority Date Predictor

## Overview
A full-stack application with a PyTorch-based machine learning system that predicts green card priority date movements for India. The project analyzes historical visa bulletin data to help applicants estimate waiting times for employment-based immigration categories, providing predictions through both a machine learning model and a user-friendly web interface.

## Features
- PyTorch-based prediction model for Indian employment-based categories
- Interactive web interface for priority date predictions
- Employment-based category predictions (EB1, EB2, EB3)
- Historical priority date movement analysis
- Movement trend visualization
- Retrogression pattern analysis
- Real-time predictions through web API
- Comprehensive data pipeline from visa bulletin scraping to prediction

The backend leverages PyTorch for deep learning-based predictions, analyzing patterns in historical visa bulletin data to generate accurate waiting time estimates. The frontend provides an intuitive interface for accessing these predictions, making the complex data analysis accessible to users.

## Tech Stack
### Frontend
- Next.js 15+
- TypeScript
- Tailwind CSS
- shadcn components

### Backend
- Python FastAPI
- Pandas for data processing
- scikit-learn for ML models
- Uvicorn for ASGI server

## Directory Structure
```
greencard-priority-predictor/
├── frontend/                    # Next.js web application
│   ├── src/
│   │   ├── app/                # Next.js App Router
│   │   │   └── api/
│   │   ├── components/         # React components
│   │   └── lib/               # Utilities and API functions
│   └── package.json
│
├── backend/                    # Python FastAPI service
│   ├── main.py                # FastAPI application
│   ├── scraper.py             # Data collection
│   ├── data/
│   │   ├── processed/         # Cleaned data
│   │   └── raw/              # Raw scraped data
│   └── requirements.txt
```

## Setup and Installation

### Backend Setup
1. Create and activate Python virtual environment
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
```

2. Install Python dependencies
```bash
pip install -r requirements.txt
```

3. Start the FastAPI server
```bash
python main.py
```
The API will be available at http://localhost:8000

### Frontend Setup
1. Install Node.js dependencies
```bash
cd frontend
npm install
```

2. Start the development server
```bash
npm run dev
```
The application will be available at http://localhost:3000

## Environment Variables

### Frontend (.env.local)
```
NEXT_PUBLIC_API_URL=http://localhost:8000
```

### Backend (.env)
```
HOST=0.0.0.0
PORT=8000
FRONTEND_URL=http://localhost:3000
```

## Development Status
- [x] Project structure setup
- [ ] Frontend UI implementation
- [ ] Backend API setup
- [ ] Frontend-Backend integration
- [ ] ML model integration
- [x] Data collection
- [ ] Data preprocessing
- [ ] Historical data analysis
- [ ] Production deployment

## Contributing
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request