#classification_model.py
from sklearn.ensemble import RandomForestClassifier

class ClassificationModel:
    def __init__(self, n_estimators=200, max_depth=None, min_samples_leaf=1, min_samples_split=2, random_state=42):

        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split,
            random_state=random_state
        )

    def train(self, X, y):
        self.model.fit(X, y)
        return self.model

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y):
        return self.model.score(X, y)

