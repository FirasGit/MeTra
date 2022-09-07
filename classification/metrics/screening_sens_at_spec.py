import sys
from sklearn import metrics
import numpy as np


def screening_sens_at_spec(y_true, y_pred):
    try:
        eps = sys.float_info.epsilon
        at_spec = 0.95
        fpr, tpr, threshes = metrics.roc_curve(
            y_true, y_pred, drop_intermediate=False)
        spec = 1 - fpr

        operating_points_with_good_spec = spec >= (at_spec - eps)
        max_tpr = tpr[operating_points_with_good_spec][-1]

        operating_point = np.argwhere(
            operating_points_with_good_spec).squeeze()[-1]
        operating_tpr = tpr[operating_point]

        assert max_tpr == operating_tpr or (np.isnan(max_tpr) and np.isnan(
            operating_tpr)), f'{max_tpr} != {operating_tpr}'
        assert max_tpr == max(tpr[operating_points_with_good_spec]) or (np.isnan(max_tpr) and max(tpr[operating_points_with_good_spec])), \
            f'{max_tpr} == {max(tpr[operating_points_with_good_spec])}'

        return max_tpr
    except:
        return 0
