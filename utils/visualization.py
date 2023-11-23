import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

def visualize_decision_trees(random_forest_model, feature_names, max_trees=3):

    num_trees = min(max_trees, len(random_forest_model.estimators_))

    for i in range(num_trees):
        tree = random_forest_model.estimators_[i]
        plt.figure(figsize=(50, 60))
        plot_tree(tree, filled=True, feature_names=feature_names, rounded=True)
        plt.title(f"Decision Tree {i}")
        plt.show()