from sklearn import metrics


def partial_roc_auc_score(y_true, y_pred):
    min_spec = 0.9
    try:
        return metrics.roc_auc_score(y_true, y_pred, max_fpr=(1 - min_spec))
    except:
        return 0.5
