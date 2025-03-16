# evaluation/metrics.py
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

def evaluate_predictions(A_true, A_pred):
    """Compute AUROC, AUPRC, and F1-score."""
    auroc = roc_auc_score(A_true.flatten(), A_pred.flatten())
    auprc = average_precision_score(A_true.flatten(), A_pred.flatten())
    f1 = f1_score(A_true.flatten(), (A_pred > 0.5).flatten())
    return auroc, auprc, f1