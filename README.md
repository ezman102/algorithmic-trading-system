# Stock Prediction with Random Forest

This project predicts stock prices using Random Forest models with technical indicators. It supports both classification (predicting price direction) and regression (predicting future price) approaches.

## Features

- Fetches historical stock data from Yahoo Finance
- Adds custom technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands, etc.)
- Implements dynamic cross-validation strategies
- Performs exhaustive feature combination search
- Generates visualizations (feature importances, decision trees, results)
- Saves trained models with performance metrics
- Produces prediction reports

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/stock-prediction-rf.git
cd stock-prediction-rf
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Execution
Run the main script:
```bash
python main.py
```

### Input Options
During execution, you'll be prompted for:
1. Stock symbol (e.g., TSLA)
2. Start and end dates (YYYY-MM-DD format)
3. Interval (1d, 1wk, 1mo)
4. Technical indicators to include
5. Prediction mode (classification or regression)

### Customizing Execution
Modify these variables in `main.py` for different defaults:
```python
stock_symbol = 'TSLA'
start_date = '2020-01-01'
end_date = '2020-12-31'
interval = '1d'
```

## Technical Indicators
The system supports these indicators:
1. SMA (10, 30 days)
2. EMA (10, 30 days)
3. RSI (14 days)
4. MACD (12/26/9 days)
5. Bollinger Bands (20 days)
6. ATR (14 days)
7. Stochastic Oscillator
8. OBV
9. Custom RSI-SMA Crossover Rule

## Outputs
The system generates:
- Trained models (in `/best_models/stock_symbol/`)
- Prediction reports (text files)
- Visualizations:
  - Feature importances
  - Decision trees
  - Classification reports
  - Regression plots
  - Target distributions
- CSV of all feature combinations (classification mode)

## File Structure
```
├── main.py                 # Main application script
├── best_models/            # Saved models
├── utils/                  # Utility modules
│   ├── data_fetcher.py     # Data retrieval
│   ├── feature_engineering.py # Technical indicators
│   ├── visualization.py    # Plotting functions
│   └── dynamic_cv_strategy.py # Cross-validation
├── requirements.txt        # Dependencies
└── README.md               # This file
```

## Dependencies
- Python 3.7+
- pandas
- numpy
- scikit-learn
- yfinance
- matplotlib
- joblib

## Contributing
Contributions are welcome! Please open an issue or submit a pull request.
