from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve, average_precision_score

def compute_metrics(y_true, y_pred_prob):
    """
    Computes precision, recall, f1, and AUPRC.
    y_pred_prob are probabilities after Sigmoid.
    """
    # Threshold predictions at 0.5
    y_pred = (y_pred_prob >= 0.5).astype(int)
    
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Calculate AUPRC
    auprc = average_precision_score(y_true, y_pred_prob)
    precisions, recalls, _ = precision_recall_curve(y_true, y_pred_prob)
    
    return precision, recall, f1, auprc, precisions, recalls
