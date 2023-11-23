import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split

def __init__(self, max_depth=5):
    self.max_depth = max_depth
    self.tree = None

def fit(self, X, y):
    self.tree = self._build_tree(X, y, depth=0)

def gini_impurity(y):
    """
    Calculate the Gini Impurity for a set of labels.
    """
    if len(y) == 0:
        return 0
    counter = Counter(y)
    impurity = 1
    for label in counter:
        prob_of_label = counter[label] / len(y)
        impurity -= prob_of_label**2
    return impurity

def information_gain(left, right, current_impurity):
    """
    Calculate the Information Gain of a split.
    """
    p = float(len(left)) / (len(left) + len(right))
    return current_impurity - p * gini_impurity(left) - (1 - p) * gini_impurity(right)

class DecisionNode:
    """
    A Decision Node asks a question.
    """
    def __init__(self, question, true_branch, false_branch):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch

class Leaf:
    """
    A Leaf node classifies data.
    """
    def __init__(self, rows):
        self.predictions = Counter(rows)

class DecisionTree:
    """
    The Decision Tree class.
    """
    def __init__(self, max_depth=5):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        if depth == self.max_depth or len(set(y)) == 1:
            return Leaf(y)

        best_gain = 0
        best_question = None
        current_impurity = gini_impurity(y)
        n_features = len(X[0])

        for col in range(n_features):
            values = set([row[col] for row in X])
            for val in values:
                question = (col, val)
                true_rows, false_rows = self._partition(X, y, question)

                if len(true_rows) == 0 or len(false_rows) == 0:
                    continue

                gain = information_gain([row[-1] for row in true_rows],
                                         [row[-1] for row in false_rows],
                                         current_impurity)

                if gain >= best_gain:
                    best_gain, best_question = gain, question

        if best_gain == 0:
            return Leaf(y)

        true_rows, false_rows = self._partition(X, y, best_question)
        true_branch = self._build_tree(true_rows, [row[-1] for row in true_rows], depth + 1)
        false_branch = self._build_tree(false_rows, [row[-1] for row in false_rows], depth + 1)

        return DecisionNode(best_question, true_branch, false_branch)

    def _partition(self, X, y, question):
        true_rows, false_rows = [], []
        for i, row in enumerate(X):
            if row[question[0]] == question[1]:
                true_rows.append(row + [y[i]])
            else:
                false_rows.append(row + [y[i]])
        return true_rows, false_rows

    def _classify(self, row, node):
        if isinstance(node, Leaf):
            return node.predictions

        # Debugging prints
        print(f"Question Index: {node.question[0]}, Row Length: {len(row)}")

        if row[node.question[0]] == node.question[1]:
            return self._classify(row, node.true_branch)
        else:
            return self._classify(row, node.false_branch)


    def predict(self, X):
        """
        Predict the class for each sample in X.
        """
        predictions = [self._classify(row, self.tree) for row in X]
        # Since predictions are returned as Counter objects, we extract the most common class
        return [max(pred, key=pred.get) for pred in predictions]



class RandomForestModel:
    """
    The Random Forest model.
    """
    def __init__(self, n_estimators=10, max_depth=5):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_estimators):
            tree = DecisionTree(max_depth=self.max_depth)
            X_sample, y_sample = self._bootstrap_sample(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def _bootstrap_sample(self, X, y):
        n_samples = len(X)
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        X_sample = [X[i] for i in indices]  # Use list comprehension
        y_sample = [y[i] for i in indices]  # Use list comprehension
        return X_sample, y_sample


    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        majority_votes = [Counter(np.argmax(row, axis=1)).most_common(1)[0][0] for row in tree_preds]
        return majority_votes

    def score(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)

import numpy as np
import pandas as pd

# Generate synthetic data
np.random.seed(0)  # For reproducibility
X = np.random.rand(100, 2)  # 100 samples, 2 features
y = np.random.randint(0, 2, 100)  # 100 binary labels

# Convert to DataFrame for easier handling
df = pd.DataFrame(X, columns=['Feature1', 'Feature2'])
df['Label'] = y

from sklearn.model_selection import train_test_split

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df[['Feature1', 'Feature2']], df['Label'], test_size=0.3)

# Convert data to list format for our custom model
X_train_list = X_train.values.tolist()
X_test_list = X_test.values.tolist()
y_train_list = y_train.tolist()
y_test_list = y_test.tolist()

# Decision Tree
decision_tree = DecisionTree(max_depth=3)
decision_tree.fit(X_train_list, y_train_list)

# Random Forest
random_forest = RandomForestModel(n_estimators=5, max_depth=3)
random_forest.fit(X_train_list, y_train_list)

# Predictions from Decision Tree
dt_predictions = decision_tree.predict(X_test_list)
# Since the predictions are in the form of Counter objects, we need to extract the most common class
dt_predictions = [max(pred, key=pred.get) for pred in dt_predictions]

# Predictions from Random Forest
rf_predictions = random_forest.predict(X_test_list)

# Evaluate
dt_accuracy = np.mean(np.array(dt_predictions) == y_test_list)
rf_accuracy = np.mean(np.array(rf_predictions) == y_test_list)

print(f"Decision Tree Accuracy: {dt_accuracy}")
print(f"Random Forest Accuracy: {rf_accuracy}")
