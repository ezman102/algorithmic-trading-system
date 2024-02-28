from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd


class RandomForestModel:
    def __init__(self, n_estimators=200, max_depth=None, min_samples_leaf=1, min_samples_split=2, random_state=42):
        """
        Initialize the Random Forest model with the ability to specify parameters.

        Parameters:
        n_estimators (int): The number of trees in the forest.
        max_depth (int): The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.
        min_samples_leaf (int): The minimum number of samples required to be at a leaf node.
        min_samples_split (int): The minimum number of samples required to split an internal node.
        random_state (int): A seed used by the random number generator for reproducibility.
        """
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split,
            random_state=random_state
        )

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
    # Calculate the index for splitting the data
    split_index = int(len(data) * 0.8)  # 80% for training, 20% for testing

    # Split the data in a time-ordered manner
    X_train, X_test = features[:split_index], features[split_index:]
    y_train, y_test = target[:split_index], target[split_index:]

    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # Initialize and train the model
    model = RandomForestModel()
    model.train(X_train, y_train)

    # Evaluate the model
    accuracy = model.evaluate(X_test, y_test)
    print(f"Model Accuracy: {accuracy}")
