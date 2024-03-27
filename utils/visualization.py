# visualization.py

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd 

def visualize_decision_trees(random_forest_model, feature_names, max_trees=3):
    num_trees = min(max_trees, len(random_forest_model.estimators_))

    for i in range(num_trees):
        tree = random_forest_model.estimators_[i]
        plt.figure(figsize=(20, 8))  # Adjust the figure size here
        plot_tree(tree, filled=True, feature_names=feature_names, rounded=True)
        plt.title(f"Decision Tree {i}")
        plt.show()

def visualize_classification_results(y_test, predictions):
    cm = confusion_matrix(y_test, predictions)
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

def visualize_regression_results(dates, y_test, predictions):
    dates = pd.to_datetime(dates)
    
    if isinstance(dates, pd.DatetimeIndex):
        data_to_plot = pd.DataFrame({'Actual': y_test, 'Prediction': predictions}).reset_index()
        data_to_plot.rename(columns={'index': 'Date'}, inplace=True)
    else:
        data_to_plot = pd.DataFrame({'Date': dates, 'Actual': y_test, 'Prediction': predictions})
    
    data_to_plot.sort_values(by='Date', inplace=True)
    
    plt.figure(figsize=(15, 7))
    plt.plot(data_to_plot['Date'], data_to_plot['Actual'], color='blue', label='Actual Values')
    plt.plot(data_to_plot['Date'], data_to_plot['Prediction'], color='red', linestyle='--', label='Predicted Values')
    plt.title('Actual vs Predicted Values for Regression')
    plt.xlabel('Date')
    plt.ylabel('Values')
    plt.legend()
    plt.xticks(rotation=45)  
    plt.tight_layout() 
    plt.show()


import matplotlib.pyplot as plt
import pandas as pd

import matplotlib.pyplot as plt
import pandas as pd

def visualize_feature_importances(features, feature_importances):
    # Create a pandas series with feature importances, indexed by feature names
    importances = pd.Series(feature_importances, index=features.columns)
    
    # Sort the feature importances in descending order
    importances_sorted = importances.sort_values(ascending=True)
    
    plt.figure(figsize=(10, 6))
    importances_sorted.plot(kind='barh', color='lightblue')
    plt.title('Feature Importance')
    
    # Annotate the bars with the feature importance values
    for index, value in enumerate(importances_sorted):
        plt.text(value, index, f'{value:.4f}')  # Format the value

    plt.show()

def plot_target_distribution(target_data):
    plt.figure(figsize=(10, 5))
    sns.countplot(x=target_data)
    plt.title('Distribution of Target Classes')
    plt.xlabel('Class')
    plt.ylabel('Frequency')
    plt.show()
