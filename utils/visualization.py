# visualization.py

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.metrics import confusion_matrix
import seaborn as sns

def visualize_decision_trees(random_forest_model, feature_names, max_trees=3):

    num_trees = min(max_trees, len(random_forest_model.estimators_))

    for i in range(num_trees):
        tree = random_forest_model.estimators_[i]
        plt.figure(figsize=(100, 50))
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
    plt.figure(figsize=(15, 7))
    plt.plot(dates, y_test, color='blue', label='Actual Values')
    plt.plot(dates, predictions, color='red', linestyle='--', label='Predicted Values')
    plt.title('Actual vs Predicted Values for Regression')
    plt.xlabel('Date')
    plt.ylabel('Values')
    plt.legend()
    plt.show()
