import numpy as np
from sklearn import metrics

def calculate_binary_classification_eer(ground_truths, probabilities):
    """
    Calculates the EER score between ground truths and probabilities.
    Args:
        ground_truths (np.ndarray): Ground truth probabilities.
        probabilities (np.ndarray): Probabilities.
    Returns:
        eer (float): EER score.
    """
    fpr, tpr, thresholds = metrics.roc_curve(ground_truths, probabilities, pos_label=1)
    fnr = 1 - tpr
    return fpr[np.nanargmin(np.absolute((fnr - fpr)))]


def calculate_multi_classification_auc(ground_truth, pred_logits):
    """
    Calculate EER for multi-class classification.
    Args:
        ground_truth (np.array): Ground truth binary classification.
        pred_logits (np.array): Predicted logits.
    Returns:
        eer (float): EER score.
    """
    # Todo
    pass