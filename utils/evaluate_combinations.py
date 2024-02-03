import pandas as pd
from itertools import combinations
from models.random_forest_model import RandomForestModel
from utils.backtester import Backtester
from joblib import Parallel, delayed
import pandas as pd
from itertools import combinations


def evaluate_combination(subset, data):
    features = data[list(subset)]
    features.fillna(features.mean(), inplace=True)
    target = data['target']

    # Split the data in a time-ordered manner
    split_index = int(len(features) * 0.8)  # for example, 80% for training, 20% for testing
    X_train, X_test = features.iloc[:split_index], features.iloc[split_index:]
    y_train, y_test = target.iloc[:split_index], target.iloc[split_index:]

    model = RandomForestModel(n_estimators=100, random_state=42)
    model.train(X_train, y_train)

    backtester = Backtester(pd.concat([X_test, y_test], axis=1), model)
    profit_loss = backtester.simulate_trading()

    return subset, profit_loss

def evaluate_feature_combinations_parallel(data, all_features, max_features=5):
    results = Parallel(n_jobs=-1)(delayed(evaluate_combination)(subset, data) 
                                   for r in range(1, max_features + 1) 
                                   for subset in combinations(all_features, r))
    
    best_combination, max_profit = max(results, key=lambda x: x[1])
    return best_combination, max_profit

