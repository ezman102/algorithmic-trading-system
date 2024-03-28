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
from utils.visualization import visualize_decision_trees, visualize_classification_results, visualize_regression_results, visualize_feature_importances,plot_target_distribution
from utils.dynamic_cv_strategy import dynamic_cv_strategy
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from itertools import chain, combinations
from sklearn.metrics import precision_score, recall_score, f1_score

def preprocess_data(features):
    imputer = SimpleImputer(strategy='mean')
    features_imputed = imputer.fit_transform(features)
    return pd.DataFrame(features_imputed, columns=features.columns)

def all_subsets(ss):
    return chain(*map(lambda x: combinations(ss, x), range(1, len(ss)+1)))

def capture_best_features_combinations(best_combinations):
                    columns = ['Combination', 'Accuracy', 'Prediction']
                    best_features_df = pd.DataFrame.from_records(best_combinations, columns=columns)
                    return best_features_df

def main():
    # stock_symbol = 'TSLA'
    # start_date = '2023-03-20'
    # end_date = '2024-03-27'
    # interval='1d'、
    stock_symbol = input("Enter the stock symbol (e.g., TSLA): ")
    start_date = input("Enter the start date (YYYY-MM-DD): ")
    end_date = input("Enter the end date (YYYY-MM-DD): ")
    interval = input("Enter the interval (1d, 1wk, 1mo): ")

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
        '14': ('Custom_Rule', {'type': 'Custom_Rule', 'rule': 'RSI_SMA_Crossover'}),
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
        # 'n_estimators': [100],
        # 'max_depth': [2,3],
        # 'min_samples_split': [2, 5],
        # 'min_samples_leaf': [1,2]
    }
    classification = True if mode == 'classification' else False

    # Use the dynamic_cv_strategy function to get an appropriate cross-validation strategy
    cv_strategy = dynamic_cv_strategy(target=y_train, classification=classification, n_splits=5)

    # Now, use cv_strategy in GridSearchCV
    grid_search = GridSearchCV(model, param_grid, cv=cv_strategy, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)

    print("Best Parameters:", grid_search.best_params_)
    best_model = grid_search.best_estimator_

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

                # Loop through all combinations
                # Initialize an empty list to store the performance of all combinations
                imputer = SimpleImputer(strategy='mean')

                # Initialize an empty list to store the performance of all combinations
                all_combination_performance = []
                print("Evaluating combinations. This may take a while...")
                # Loop through all combinations
                for combination in all_combinations:
                    # Select only the current combination of features
                    features_subset = features[list(combination)]
                    print(f"Evaluating combination: {' '.join(combination)}")
                    # Preprocess the data to fill NaN values
                    features_subset_imputed = imputer.fit_transform(features_subset)
                    
                    X_train_subset, X_test_subset, y_train_subset, y_test_subset = train_test_split(
                        features_subset_imputed, target, test_size=0.1, random_state=42
                    )

                    # Train the model
                    model = RandomForestClassifier()
                    # model = RandomForestClassifier(n_estimators=200, max_depth=None, min_samples_leaf=1, min_samples_split=2, random_state=42)
                    model.fit(X_train_subset, y_train_subset)

                    # Evaluate the model
                    accuracy = model.score(X_test_subset, y_test_subset)
                    predictions = model.predict(X_test_subset)
                    precision = precision_score(y_test_subset, predictions, average='macro')
                    recall = recall_score(y_test_subset, predictions, average='macro')
                    f1 = f1_score(y_test_subset, predictions, average='macro')
                    last_features = original_data.iloc[-1:][list(combination)].ffill().bfill().values
                    next_day_prediction = model.predict(last_features)[0]

                    # Store the combination, accuracy, and prediction
                    
                    combination_performance = {
                        'Combination': ' '.join(combination),
                        'Accuracy': accuracy,
                        'Precision': precision,
                        'Recall': recall,
                        'F1 Score': f1,
                        'Next_Day_Prediction': next_day_prediction
                    }

                    # Update the best model if the current one is better
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_combination = combination
                        best_model = model

                    all_combination_performance.append(combination_performance)

                # Convert the list of dictionaries to a DataFrame
                all_combination_performance_df = pd.DataFrame(all_combination_performance)

                # Save the DataFrame to a CSV file
                csv_file_path = os.path.join(models_dir, 'all_combination_performance.csv')
                all_combination_performance_df.to_csv(csv_file_path, index=False)
                print(f"All combinations performance saved to {csv_file_path}")


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
                plot_target_distribution(data['target_class'])

                # Make predictions with the best model
                # Before making predictions, ensure features are correctly selected and ordered
                features_for_prediction = [feature for feature in list(best_combination) if feature in X_test.columns]

                # Preprocess the features for prediction to ensure they match training data format
                X_test_best_preprocessed = preprocess_data(X_test[features_for_prediction])

                # Make predictions with the best model
                best_predictions = best_model.predict(X_test_best_preprocessed)

                # Calculate and print metrics
                print(f"Best Model Accuracy: {accuracy_score(y_test, best_predictions)}")
                print(f"Best Model Precision: {precision_score(y_test, best_predictions, average='macro')}")
                print(f"Best Model Recall: {recall_score(y_test, best_predictions, average='macro')}")
                print(f"Best Model F1 Score: {f1_score(y_test, best_predictions, average='macro')}")

                # Visualize the classification results for the best model
                visualize_classification_results(y_test, best_predictions)
                # Ensure your visualization functions are compatible with the specifics of your dataset and model

                # Visualize the decision trees for the best model, if applicable
                visualize_decision_trees(best_model, list(best_combination), max_trees=1)
                # Adjust the call based on your visualization function's requirements
                
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