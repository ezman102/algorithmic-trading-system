# visualization.py

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd 
import seaborn as sns
import numpy as np
import os


def visualize_decision_trees(random_forest_model, feature_names, stock_symbol, start_date, end_date, max_trees=3, prefix='', output_folder=''):
    num_trees = min(max_trees, len(random_forest_model.estimators_))
    for i in range(num_trees):
        tree = random_forest_model.estimators_[i]
        filename = os.path.join(output_folder, f"{stock_symbol}_{start_date}_to_{end_date}_{prefix}_decision_tree_{i}.png")
        plt.figure(figsize=(20, 8))  # Adjust the figure size here
        plot_tree(tree, filled=True, feature_names=feature_names, rounded=True)
        plt.title(f"Decision Tree {i} for {stock_symbol} from {start_date} to {end_date}")
        plt.savefig(filename)
        plt.show()

def visualize_classification_results(y_test, predictions, stock_symbol, start_date, end_date, output_folder=''):
    cm = confusion_matrix(y_test, predictions)
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title(f"Confusion Matrix for {stock_symbol} from {start_date} to {end_date}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    filename = os.path.join(output_folder, f'{stock_symbol}_{start_date}_to_{end_date}_classification_confusion_matrix.png')
    plt.savefig(filename)
    plt.show()

def visualize_regression_results(dates, y_test, predictions, stock_symbol, start_date, end_date, output_folder=''):
    dates = pd.to_datetime(dates)
    if isinstance(dates, pd.DatetimeIndex):
        data_to_plot = pd.DataFrame({'Actual': y_test, 'Prediction': predictions}).reset_index()
        data_to_plot.rename(columns={'index': 'Date'}, inplace=True)
    else:
        data_to_plot = pd.DataFrame({'Date': dates, 'Actual': y_test, 'Prediction': predictions})
    data_to_plot.sort_values(by='Date', inplace=True)
    
    plt.figure(figsize=(15, 7))
    actual_line, = plt.plot(data_to_plot['Date'], data_to_plot['Actual'], color='blue', label='Actual Values')
    predicted_line, = plt.plot(data_to_plot['Date'], data_to_plot['Prediction'], color='red', linestyle='--', label='Predicted Values')
    differences = np.abs(data_to_plot['Actual'] - data_to_plot['Prediction'])
    threshold = differences.quantile(0.7)  
    
    for i in range(len(data_to_plot)):
        if differences.iloc[i] > threshold:
            plt.text(data_to_plot['Date'].iloc[i], data_to_plot['Actual'].iloc[i], f"{data_to_plot['Actual'].iloc[i]:.2f}", fontsize=8, verticalalignment='bottom', alpha=0.7)
            plt.text(data_to_plot['Date'].iloc[i], data_to_plot['Prediction'].iloc[i], f"{data_to_plot['Prediction'].iloc[i]:.2f}", fontsize=8, verticalalignment='top', color='red', alpha=0.7)
    
    plt.title(f'Actual vs Predicted Values for {stock_symbol} from {start_date} to {end_date}')
    plt.xlabel('Date')
    plt.ylabel('Values')
    plt.legend(handles=[actual_line, predicted_line])
    plt.xticks(rotation=45)
    plt.tight_layout()
    filename = os.path.join(output_folder, f'{stock_symbol}_{start_date}_to_{end_date}_regression_actual_vs_predicted.png')
    plt.savefig(filename)
    plt.show()

def visualize_feature_importances(feature_names, feature_importances, stock_symbol, start_date, end_date, prefix='', output_folder=''):
    importances = pd.Series(feature_importances, index=feature_names)
    importances_sorted = importances.sort_values(ascending=True)
    plt.figure(figsize=(10, 6))
    ax = importances_sorted.plot(kind='barh', color='lightblue')
    plt.title(f'Feature Importance for {stock_symbol} from {start_date} to {end_date}')
    for index, value in enumerate(importances_sorted):
        plt.text(value + ax.get_xlim()[1]*0.01, index, f'{value:.4f}', va='center') 
    plt.subplots_adjust(left=0.15, right=0.95)
    filename = os.path.join(output_folder, f"{stock_symbol}_{start_date}_to_{end_date}_{prefix}_feature_importance.png")
    plt.savefig(filename)
    plt.show()
    

def plot_target_distribution(target_data, stock_symbol, start_date, end_date, output_folder=''):
    plt.figure(figsize=(10, 5))
    # Create countplot
    ax = sns.countplot(x=target_data, hue=target_data, palette=['red', 'blue'], legend=False)
    plt.title(f'Distribution of Target Classes for {stock_symbol} from {start_date} to {end_date}')
    plt.xlabel('Class')
    plt.ylabel('Frequency')
    for p in ax.patches:
        ax.text(p.get_x() + p.get_width() / 2., p.get_height(), '%d' % int(p.get_height()),
                fontsize=12, color='black', ha='center', va='bottom')
    filename = os.path.join(output_folder, f'{stock_symbol}_{start_date}_to_{end_date}_classification_target_distribution.png')
    plt.savefig(filename)
    plt.show()