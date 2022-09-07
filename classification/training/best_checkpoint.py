import os
import re
import numpy as np


def get_best_checkpoint(checkpoint_dir, metric='Val_mean_auc', mode='max'):
    resume_from_checkpoint = None
    if os.path.exists(checkpoint_dir):
        checkpoint_files = [os.path.join(checkpoint_dir, file) for file in os.listdir(checkpoint_dir)
                            if os.path.isfile(os.path.join(checkpoint_dir, file))]
        if checkpoint_files:
            resume_from_checkpoint = _get_filename_with_best_score(
                checkpoint_files, metric, mode)
    print("Resume from checkpoint:",  resume_from_checkpoint)

    return resume_from_checkpoint


def _get_filename_with_best_score(checkpoint_files, metric, mode):
    # Get value with highest metric score
    scores = [_find_metric_value(metric, file) for file in checkpoint_files]
    try:
        if mode == 'max':
            # ignores nan when looking for argmax
            best_file_idx = np.nanargmax(scores)
        elif mode == 'min':
            best_file_idx = np.nanargmin(scores)
        return checkpoint_files[best_file_idx]
    except ValueError:
        # np.nanargmax() and np.nanargmin() raise ValueError if all values are nan
        return None


def _find_metric_value(metric, file):
    search = re.search(rf'{re.escape(metric)}=(.{{5}})', file, re.IGNORECASE)
    return np.float(search.group(1)) if search else np.nan
