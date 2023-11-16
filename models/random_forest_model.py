from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

class RandomForestModel:
    def __init__(self, n_estimators=500, random_state=42):
        """
        Initialize the Random Forest model.

        Parameters:
        n_estimators (int): The number of trees in the forest.
        random_state (int): A seed used by the random number generator.
        """
        self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)

    def train(self, X, y):
        """
        Train the Random Forest model.

        Parameters:
        X (pandas.DataFrame): Features for training.
        y (pandas.Series): Target variable.

        Returns:
        The trained model.
        """
        self.model.fit(X, y)
        return self.model

    def predict(self, X):
        """
        Make predictions using the trained model.

        Parameters:
        X (pandas.DataFrame): Features for making predictions.

        Returns:
        numpy.array: Predicted values.
        """
        return self.model.predict(X)

    def evaluate(self, X, y):
        """
        Evaluate the performance of the model.

        Parameters:
        X (pandas.DataFrame): Features for evaluation.
        y (pandas.Series): True target values.

        Returns:
        float: The accuracy of the model.
        """
        return self.model.score(X, y)

# Example usage

if __name__ == "__main__":
    # Load the data
    data = pd.read_csv("data/enhanced_stock_data.csv")

    # Calculate the Price Movement Direction as the target
    data['target'] = (data['Close'].shift(-1) > data['Close']).astype(int)

    # Drop rows with NaN values
    data.dropna(inplace=True)

    # Define features and target variable
    features = data.drop('target', axis=1)
    target = data['target']

    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # Initialize and train the model
    model = RandomForestModel()
    model.train(X_train, y_train)

    # Evaluate the model
    accuracy = model.evaluate(X_test, y_test)
    print(f"Model Accuracy: {accuracy}")
