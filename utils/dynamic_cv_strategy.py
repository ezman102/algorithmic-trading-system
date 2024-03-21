from sklearn.model_selection import KFold, StratifiedKFold

def dynamic_cv_strategy(target, classification=True, n_splits=5):
    """
    Determines the cross-validation strategy based on the task type and dataset characteristics.
    
    Args:
    - target (pd.Series): The target variable.
    - classification (bool): Indicates if the task is classification. Default is True.
    - n_splits (int): The number of splits/folds for cross-validation. Default is 5.
    
    Returns:
    - An initialized cross-validation object.
    """
    # For classification tasks, use StratifiedKFold to preserve the percentage of samples for each class.
    if classification:
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    # For regression tasks, use KFold.
    else:
        return KFold(n_splits=n_splits, shuffle=True, random_state=42)