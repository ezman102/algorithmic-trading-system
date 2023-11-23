import random

# Sample data with two features and binary classification
data = [
    (1.2, 2.3, 0),
    (2.8, 3.4, 0),
    (3.5, 1.9, 1),
    (6.1, 5.0, 1),

]

# Define the number of decision trees in the forest
num_trees = 3

# Define the number of features to consider for each split
num_features = 2

# Define the depth of each decision tree
max_depth = 3

# Define the random forest
forest = []

# Create random decision trees
for _ in range(num_trees):
    tree = {}  # Each tree is represented as a dictionary
    # Randomly select features for this tree
    selected_features = random.sample(range(len(data[0]) - 1), num_features)
    
    # Generate a random decision tree with a limited depth
    def build_tree(data, depth):
        if depth >= max_depth or len(data) == 0:
            # Create a leaf node with the majority class among the remaining data
            if len(data) == 0:
                return {'leaf': True, 'class': 0}  # Assign a default class if no data points are left
            return {'leaf': True, 'class': max(set([d[-1] for d in data]), key=[d[-1] for d in data].count)}
        
        feature = random.choice(selected_features)
        threshold = random.uniform(min([d[feature] for d in data]), max([d[feature] for d in data]))
        
        left_data = [d for d in data if d[feature] <= threshold]
        right_data = [d for d in data if d[feature] > threshold]
        
        # Recursively build left and right subtrees
        tree = {
            'feature': feature,
            'threshold': threshold,
            'left': build_tree(left_data, depth + 1),
            'right': build_tree(right_data, depth + 1),
        }
        
        return tree

    tree['tree'] = build_tree(data, 0)
    forest.append(tree)

# Function to print the structure of a decision tree
def print_tree(tree, indent=0):
    if not tree:
        return
    if 'leaf' in tree:
        print("  " * indent + "Leaf Node - Class:", tree['class'])
    else:
        print("  " * indent + f"Feature {tree['feature']} <= {tree['threshold']}")
        print_tree(tree['left'], indent + 1)
        print_tree(tree['right'], indent + 1)

# Print the structure of each tree in the forest
for idx, tree in enumerate(forest):
    print(f"Tree {idx + 1} Structure:")
    print_tree(tree['tree'])
    print()

# Random forest prediction function
def predict(forest, sample):
    votes = [0, 0]
    for tree in forest:
        prediction = None
        node = tree['tree']
        while 'leaf' not in node:
            if sample[node['feature']] <= node['threshold']:
                node = node['left']
            else:
                node = node['right']
        votes[node['class']] += 1
    return votes.index(max(votes))


new_sample = (4.0, 3.5)
prediction = predict(forest, new_sample)
print("Predicted class:", prediction)

def print_tree(tree, indent=0):
    if not tree:
        return
    if 'leaf' in tree:
        print("  " * indent + "Leaf Node - Class:", tree['class'])
    else:
        print("  " * indent + f"Feature {tree['feature']} <= {tree['threshold']}")
        print_tree(tree['left'], indent + 1)
        print_tree(tree['right'], indent + 1)

for idx, tree in enumerate(forest):
    print(f"Tree {idx + 1} Structure:")
    print_tree(tree['tree'])
    print()
