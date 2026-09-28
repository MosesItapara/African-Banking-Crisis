from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

def classification_metrics(y_true, y_pred) -> dict:
    """Return a flat dict of weighted precisio/recall/f1 and accuracy"""
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    metrics = {k: report["weighted avg"][k] for k in ["precision", "recall", "f1-score"]}
    metrics["accuracy"] = report["accuracy"]
    return metrics