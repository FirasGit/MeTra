from sklearn import metrics


def roc_auc_score(y_trues, y_preds):
    try:
        return metrics.roc_auc_score(y_trues, y_preds)
    except:
        return 0.5
