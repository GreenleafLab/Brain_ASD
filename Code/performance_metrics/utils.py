import numpy as np

def remove_ambiguous_peaks(predictions, true_y,ambig_val=np.nan): 
    indices_to_remove = np.nonzero(np.isnan(true_y))
    true_y_filtered = np.delete(true_y, indices_to_remove)
    predictions_filtered = np.delete(predictions, indices_to_remove)
    return predictions_filtered, true_y_filtered
