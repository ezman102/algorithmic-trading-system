# utils/__init__.py

# For explicit module import when using 'from utils import *'
__all__ = ['data_fetcher', 'feature_engineering', 'backtester']

# import classes/functions from modules to make them
# directly accessible from the package level

from .data_fetcher import fetch_data
from .feature_engineering import add_technical_indicators, define_target_variable
from .backtester import Backtester
