from sklearn import metrics


def calculate_binary_classification_auc(ground_truth, probabilities):
    """
    Calculate AUC for binary classification.
    Args:
        ground_truth (np.array): Ground truth binary classification.
        probabilities (np.array): Predicted probabilities.
    Returns:
        auc (float): AUC.
    """
    fpr, tpr, thresholds = metrics.roc_curve(ground_truth, probabilities, pos_label=1)
    return metrics.auc(fpr, tpr)


def calculate_multi_classification_auc(ground_truth, pred_logits):
    """
    Calculate AUC for multi-class classification.
    Args:
        ground_truth (np.array): Ground truth binary classification.
        pred_logits (np.array): Predicted logits.
    Returns:
        auc (float): AUC.
    """
    # Todo
    pass
