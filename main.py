#main.py
import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
utils_dir = os.path.join(script_dir, 'utils')
sys.path.append(utils_dir)

models_dir = os.path.join(script_dir, 'best_models')
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
from sklearn.impute import SimpleImputer
import numpy as np
import joblib
import pandas as pd
from utils.data_fetcher import fetch_data
from utils.feature_engineering import add_technical_indicators, define_target_variable
from utils.visualization import visualize_decision_trees, visualize_classification_results, visualize_regression_results, visualize_feature_importances
from utils.dynamic_cv_strategy import dynamic_cv_strategy
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from itertools import chain, combinations

def preprocess_data(features):
    imputer = SimpleImputer(strategy='mean') 
    return imputer.fit_transform(features)

def all_subsets(ss):
    return chain(*map(lambda x: combinations(ss, x), range(1, len(ss)+1)))

def main():
    stock_symbol = 'TSLA'
    start_date = '2023-03-20'
    end_date = '2024-03-20'
    interval='1d'

    print("Fetching data...")
    data = fetch_data(stock_symbol, start_date, end_date, interval='1d')

    if isinstance(data.index, pd.DatetimeIndex):
        dates = data.index
    else:
        dates = pd.to_datetime(data.index)

    available_indicators = {
        '1': ('SMA_10', {'name': 'SMA_10', 'type': 'SMA', 'window': 10}),
        '2': ('SMA_30', {'name': 'SMA_30', 'type': 'SMA', 'window': 30}),
        '3': ('EMA_10', {'name': 'EMA_10', 'type': 'EMA', 'window': 10}),
        '4': ('EMA_30', {'name': 'EMA_30', 'type': 'EMA', 'window': 30}),
        '5': ('RSI_14', {'name': 'RSI_14', 'type': 'RSI', 'window': 14}),
        '6': ('MACD', {'name': 'MACD', 'type': 'MACD', 'short_window': 12, 'long_window': 26, 'signal_window': 9}),
        '7': ('MACD_Signal', {'name': 'MACD_Signal', 'type': 'MACD', 'short_window': 12, 'long_window': 26, 'signal_window': 9}),
        '8': ('MACD_Histogram', {'name': 'MACD_Histogram', 'type': 'MACD', 'short_window': 12, 'long_window': 26, 'signal_window': 9}),
        '9': ('BB_Upper', {'name': 'BB_Upper', 'type': 'BB', 'window': 20}),
        '10': ('BB_Lower', {'name': 'BB_Lower', 'type': 'BB', 'window': 20}),
        '11': ('ATR', {'name': 'ATR', 'type': 'ATR', 'window': 14}),
        '12': ('Stochastic_Oscillator', {'name': 'Stochastic_Oscillator', 'type': 'Stochastic', 'window': 14}),
        '13': ('OBV', {'name': 'OBV', 'type': 'OBV'}),
    }

    print("Available indicators:")
    for number, (name, _) in available_indicators.items():
        print(f"{number}: {name}")

    indicator_numbers = input("Enter the numbers of the indicators you want to add (separated by space): ")
    selected_indicator_numbers = indicator_numbers.split()

    selected_indicators = []
    for number in selected_indicator_numbers:
        if number in available_indicators:
            _, indicator_params = available_indicators[number]
            selected_indicators.append(indicator_params)
        else:
            print(f"Indicator number {number} is not valid. It will be skipped.")

    if not selected_indicators:
        print("No valid indicators selected. Exiting...")
        sys.exit(0)

    print("Adding selected technical indicators...")
    data = add_technical_indicators(data, selected_indicators, drop_original=True)

    original_data = data.copy()
    print("Select mode:")
    print("1. Classification")
    print("2. Regression")
    choice = input("Enter choice (1/2): ")
    
    if choice == '1':
        mode = 'classification'
    elif choice == '2':
        mode = 'regression'
    else:
        print("Invalid choice. Exiting...")
        sys.exit(1)

    if mode == 'classification':
        model = RandomForestClassifier(random_state=42)
        data = define_target_variable(data, 'target_class', 1)  # for binary classification
        target_column = 'target_class'
    elif mode == 'regression':
        model = RandomForestRegressor(random_state=42)
        data = define_target_variable(data, 'target_reg', 0, is_regression=True)  # for continuous target
        target_column = 'target_reg'
    else:
        print("Invalid mode selected. Exiting...")
        return
    
    data.drop(columns=['Close'], inplace=True, errors='ignore')

    features = data.drop([target_column], axis=1)
    target = data[target_column]

    preprocessed_features = preprocess_data(features)

    X_train, X_test, y_train, y_test = train_test_split(preprocessed_features, target, test_size=0.1, random_state=42)
    

    param_grid = {
        'n_estimators': [100, 200, 500, 1000],
        'max_depth': [None, 2, 4, 5, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
        # 'n_estimators': [100, 200],
        # 'max_depth': [2],
        # 'min_samples_split': [2, 5, 10],
        # 'min_samples_leaf': [1, 2, 4]
    }
    classification = True if mode == 'classification' else False

    # Use the dynamic_cv_strategy function to get an appropriate cross-validation strategy
    cv_strategy = dynamic_cv_strategy(target=y_train, classification=classification, n_splits=5)

    # Now, use cv_strategy in GridSearchCV
    grid_search = GridSearchCV(model, param_grid, cv=cv_strategy, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)

    print("Best Parameters:", grid_search.best_params_)
    best_model = grid_search.best_estimator_

    # Assuming best_model is your trained model
    # And assuming 'features' is your DataFrame of feature data
    feature_importances = best_model.feature_importances_

    # Call the visualization function with the DataFrame and the importances
    visualize_feature_importances(features, feature_importances)

    predictions = best_model.predict(X_test)
    latest_data = original_data.iloc[-1:][features.columns].ffill().bfill().values


    if mode == 'classification':
                # Create all possible combinations of features
                all_features = list(features.columns)
                all_combinations = list(all_subsets(all_features))

                best_accuracy = 0
                best_combination = None
                best_model = None

                for combination in all_combinations:
                    # Select only the current combination of features
                    X_train_subset = X_train[:, features.columns.isin(combination)]
                    X_test_subset = X_test[:, features.columns.isin(combination)]

                    model = RandomForestClassifier(random_state=42)
                    model.fit(X_train_subset, y_train)

                    # Evaluate the model
                    accuracy = model.score(X_test_subset, y_test)
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_combination = combination
                        best_model = model
                
                # Save the best model
                model_filename = os.path.join(
                models_dir,
                f"{stock_symbol}_from_{start_date}_to_{end_date}_accuracy_{best_accuracy:.4f}.joblib"
                )

                joblib.dump(best_model, model_filename)
                print(f"Best combination of features: {best_combination}")
                print(f"Best accuracy: {best_accuracy}")

                latest_features = original_data.iloc[-1:][list(best_combination)].ffill().bfill().values

                next_day_prediction = best_model.predict(latest_features)

                # Output the prediction for the next day
                print(f"Predicted value for the next day: {next_day_prediction[0]}")

                print(f"Model saved as {model_filename}")
                visualize_classification_results(y_test, predictions)
                visualize_decision_trees(best_model, features.columns, max_trees=1)



    elif mode == 'regression':
        # Calculate mean squared error, mean absolute error, and root mean squared error
        predictions = best_model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mse)

        # Print out the performance metrics
        print(f"Regression MSE: {mse}")
        print(f"Regression MAE: {mae}")
        print(f"Regression RMSE: {rmse}")

        # Predict the value for the next day using the latest data
        next_day_prediction = best_model.predict(latest_data)
        print(f"Predicted value for the next day: {next_day_prediction[0]}")

        # Compile the prediction with dates into a DataFrame
        predictions_with_dates = pd.DataFrame({
            'Date': dates[-len(predictions):].tolist(),
            'Actual': y_test.tolist(),
            'Prediction': predictions.tolist()
        })

        # Print actual values vs predictions
        print(y_test)
        print(predictions)

        # Define the model filename with stock symbol, date range, and RMSE
        model_filename = os.path.join(
            models_dir,
            f"{stock_symbol}_regression_{start_date}_to_{end_date}_RMSE_{rmse:.4f}.joblib"
        )

        # Save the regression model to the specified path
        joblib.dump(best_model, model_filename)
        print(f"Model saved as {model_filename}")

        # Visualize the regression results and decision trees
        visualize_regression_results(y_test.index, y_test, predictions)
        visualize_decision_trees(best_model, features.columns, max_trees=1)

if __name__ == "__main__":
    main()