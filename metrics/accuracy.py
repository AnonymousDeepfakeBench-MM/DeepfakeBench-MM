import numpy as np
from sklearn.metrics import accuracy_score


def calculate_binary_classification_accuracy(ground_truths, probabilities):
    """
    Calculates the accuracy of the binary classification task.
    Args:
        ground_truths (np.ndarray): Ground truth labels.
        probabilities (np.ndarray): Predicted probabilities.
    Returns:
        accuracy (float): Accuracy of the binary classification task.
    """
    if probabilities.max() > 1 or probabilities.min() < 0:
        raise ValueError('y_pred must be between 0 and 1. ')

    pred_labels = (probabilities > 0.5).astype(int)
    return accuracy_score(ground_truths, pred_labels)


def calculate_multi_classification_accuracy(ground_truths, pred_logits):
    """
    Calculates the accuracy of the multi-class classification task.
    Args:
        ground_truths (np.ndarray): Ground truth labels.
        pred_logits (np.ndarray): Predicted logits.
    Returns:
        accuracy (float): Accuracy of the multi-class classification task.
    """
    pred_labels = np.argmax(pred_logits, axis=1)
    return accuracy_score(ground_truths, pred_labels)