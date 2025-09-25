from sklearn import metrics

def calculate_binary_classification_average_precision(ground_truths, probabilities):
    """
    Calculate binary classification average precision.
    Args:
        ground_truths (np.ndarray): Ground truth labels.
        probabilities (np.ndarray): Predicted probabilities.
    Returns:
        average_precision (float): Binary classification average precision.
    """
    return metrics.average_precision_score(ground_truths, probabilities)