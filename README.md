# Greencard Priority Date Predictor

## Overview
A PyTorch-based machine learning system that predicts green card priority date movements for India. The project analyzes historical visa bulletin data to help applicants estimate waiting times for both employment-based and family-based immigration categories.

## Features
- Historical priority date movement analysis
- Employment-based category predictions (EB1, EB2, EB3)
- Family-based category predictions (F1, F2A, F2B, F3, F4)
- Movement trend visualization
- Retrogression pattern analysis

## Directory Structure
```
greencard-priority-predictor/
├── data/
│   ├── raw/                # Original scraped visa bulletin data
│   └── processed/          # Cleaned and formatted data
├── src/
│   ├── data/              
│   │   ├── scraper.py     # Visa bulletin scraping script
│   │   └── processor.py   # Data cleaning and preprocessing
│   ├── models/            
│   │   ├── dataset.py     # PyTorch dataset classes
│   │   ├── network.py     # Model architecture
│   │   └── train.py       # Training pipeline
│   └── utils/             # Helper functions
├── notebooks/             # Analysis and experimentation
├── tests/                 # Unit tests
└── requirements.txt       # Dependencies
```

## Setup
1. Create and activate virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

## Data Collection
```bash
python src/data/scraper.py
```

## Model Training
```bash
python src/models/train.py
```

## Project Status
- [x] Data collection
- [ ] Data preprocessing
- [ ] Model development
- [ ] Training pipeline
- [ ] Evaluation metrics
- [ ] Prediction interface

## Tech Stack
- Python 3.8+
- PyTorch
- pandas
- scikit-learn
- matplotlib/seaborn

## License
MIT License

## Contributing
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request