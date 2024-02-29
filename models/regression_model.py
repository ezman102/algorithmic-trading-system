# regression_model.py
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

class RegressionModel:
    def __init__(self, n_estimators=100, max_depth=None, random_state=42):
        """
        Initializes the Random Forest Regression model with customizable parameters.

        Parameters:
        n_estimators (int): Number of trees in the forest.
        max_depth (int): Maximum depth of the tree.
        random_state (int): Seed used by the random number generator.
        """
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )
    
    def train(self, X_train, y_train):
        """
        Trains the regression model using the provided features and target.

        Parameters:
        X_train (pd.DataFrame): Training data features.
        y_train (pd.Series): Target variable for training data.
        """
        self.model.fit(X_train, y_train)
    
    def predict(self, X_test):
        """
        Makes predictions using the trained regression model.

        Parameters:
        X_test (pd.DataFrame): Test data features.

        Returns:
        np.array: Predicted values.
        """
        return self.model.predict(X_test)
    
    def evaluate(self, X_test, y_test):
        """
        Evaluates the performance of the regression model on a test set.

        Parameters:
        X_test (pd.DataFrame): Test data features.
        y_test (pd.Series): True target values for the test data.

        Returns:
        dict: A dictionary containing the model's performance metrics.
        """
        predictions = self.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        return {"Mean Squared Error": mse, "R^2 Score": r2}

