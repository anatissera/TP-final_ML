import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve, auc

def compute_roc(y_true, scores):
    """
    Compute ROC curve and AUC.
    Returns fpr, tpr, thresholds, roc_auc
    """
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, thresholds, roc_auc


def compute_pr(y_true, scores):
    """
    Compute Precision-Recall curve and AUC.
    Returns precision, recall, thresholds, pr_auc
    """
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    pr_auc = auc(recall, precision)
    return precision, recall, thresholds, pr_auc


def optimal_threshold(fpr, tpr, thresholds):
    """
    Compute optimal threshold using Youden's J statistic.
    """
    j_scores = tpr - fpr
    idx = np.argmax(j_scores)
    return thresholds[idx]
