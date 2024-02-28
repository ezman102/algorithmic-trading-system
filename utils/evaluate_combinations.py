import pandas as pd
from itertools import combinations
from joblib import Parallel, delayed
from models.random_forest_model import RandomForestModel
from utils.backtester import Backtester

def evaluate_combination(subset, data):
    """
    Evaluate a specific combination of features for its performance.

    Parameters:
    subset (tuple): A combination of feature names.
    data (pd.DataFrame): The dataset containing features and the target variable.

    Returns:
    tuple: The subset of features evaluated and the profit/loss result.
    """
    # Select the features based on the subset
    features = data[list(subset)]
    # Fill missing values
    features.fillna(features.mean(), inplace=True)
    target = data['target']

    # Define the split index for training and testing sets
    split_index = int(len(features) * 0.9)
    X_train, X_test = features.iloc[:split_index], features.iloc[split_index:]
    y_train, y_test = target.iloc[:split_index], target.iloc[split_index:]

    # Initialize and train the model
    model = RandomForestModel(n_estimators=200, max_depth=20, min_samples_leaf=2, min_samples_split=5, random_state=42)
    model.train(X_train, y_train)

    # Initialize the backtester with the test set and the trained model
    backtester = Backtester(pd.concat([X_test, y_test], axis=1), model)
    profit_loss = backtester.simulate_trading()

    return subset, profit_loss

def evaluate_feature_combinations_parallel(data, all_features, max_features=5):
    """
    Evaluate all possible combinations of features in parallel and find the best one.

    Parameters:
    data (pd.DataFrame): The dataset containing features and the target variable.
    all_features (list): List of all feature names to consider for combinations.
    max_features (int): The maximum number of features to include in a combination.

    Returns:
    tuple: The best feature combination and its associated profit/loss.
    """
    # Evaluate all combinations of features in parallel using joblib
    results = Parallel(n_jobs=-1)(
        delayed(evaluate_combination)(subset, data) 
        for r in range(1, max_features + 1)
        for subset in combinations(all_features, r)
    )

    # Find the combination with the maximum profit/loss
    best_combination, max_profit = max(results, key=lambda x: x[1])

    return best_combination, max_profit

# # Example usage
# if __name__ == "__main__":
#     # Load your data
#     data = pd.read_csv("your_data.csv")
    
#     # Define all possible features to consider
#     all_features = ['feature1', 'feature2', 'feature3', 'feature4']  # Add your actual feature names here

#     # Evaluate feature combinations
#     best_combination, max_profit = evaluate_feature_combinations_parallel(data, all_features)
#     print(f"Best feature combination: {best_combination}")
#     print(f"Maximum Profit/Loss: {max_profit}")
