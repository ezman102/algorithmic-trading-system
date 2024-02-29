#classification_model.py
from sklearn.ensemble import RandomForestClassifier

class ClassificationModel:
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

