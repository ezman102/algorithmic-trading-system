# import matplotlib.pyplot as plt
# from sklearn.tree import plot_tree

# def visualize_decision_trees(random_forest_model, feature_names, max_trees=3):
#     # Ensure we don't try to visualize more trees than exist
#     num_trees = min(max_trees, len(random_forest_model.estimators_))

#     for i in range(num_trees):
#         tree = random_forest_model.estimators_[i]
#         plt.figure(figsize=(20, 10))
#         plot_tree(tree, filled=True, feature_names=feature_names, rounded=True)
#         plt.title(f"Decision Tree {i}")
#         plt.show()

# Check if dtreeviz is installed
try:
    from dtreeviz.trees import dtreeviz
    dtreeviz_installed = True
except ImportError:
    dtreeviz_installed = False
    from sklearn.tree import plot_tree
    import matplotlib.pyplot as plt

def visualize_decision_tree(tree, feature_names, class_names=None, target_name="target"):
    if dtreeviz_installed:
        viz = dtreeviz(tree, 
                       x_data=features.values,
                       y_data=target,
                       target_name=target_name,
                       feature_names=feature_names,
                       class_names=class_names)
        return viz
    else:
        # Fallback to plot_tree if dtreeviz is not available
        plt.figure(figsize=(20,10))
        plot_tree(tree, filled=True, feature_names=feature_names, rounded=True)
        plt.show()




