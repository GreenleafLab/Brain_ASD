import tensorflow as tf


pseudocount=0.01


def contingency_table(y, z):
    """Note:  if y and z are not rounded to 0 or 1, they are ignored
    """
    y = tf.cast(tf.round(y), tf.floatx())
    z = tf.cast(tf.round(z), tf.floatx())
    
    def count_matches(y, z):
        return tf.sum(tf.cast(y, tf.floatx()) * tf.cast(z, tf.floatx()))
    
    ones = tf.ones_like(y)
    zeros = tf.zeros_like(y)
    y_ones = tf.equal(y, ones)
    y_zeros = tf.equal(y, zeros)
    z_ones = tf.equal(z, ones)
    z_zeros = tf.equal(z, zeros)
    
    tp = count_matches(y_ones, z_ones)
    tn = count_matches(y_zeros, z_zeros)
    fp = count_matches(y_zeros, z_ones)
    fn = count_matches(y_ones, z_zeros)
    return (tp, tn, fp, fn)

def recall(y, z):
    """True positive rate `tp / (tp + fn)`
    """
    tp, tn, fp, fn = contingency_table(y, z)
    return tp / (tp + fn+pseudocount)

def tpr(y, z):
    """
    True positive rate `tp / (tp + fn)`
    """
    tp, tn, fp, fn = contingency_table(y, z)
    return tp / (tp + fn)

def tnr(y, z):
    """
    True negative rate `tn / (tn + fp)`
    """
    tp, tn, fp, fn = contingency_table(y, z)
    return tn / (tn + fp)

def specificity(y, z):
    """True negative rate `tn / (tn + fp)`
    """
    tp, tn, fp, fn = contingency_table(y, z)
    return tn / (tn + fp+pseudocount)


def fpr(y, z):
    """False positive rate `fp / (fp + tn)`
    """
    tp, tn, fp, fn = contingency_table(y, z)
    return fp / (fp + tn+pseudocount)


def fnr(y, z):
    """False negative rate `fn / (fn + tp)`
    """
    tp, tn, fp, fn = contingency_table(y, z)
    return fn / (fn + tp+pseudocount)


def precision(y, z):
    """Precision `tp / (tp + fp)`
    """
    tp, tn, fp, fn = contingency_table(y, z)
    return tp / (tp + fp+pseudocount)


def fdr(y, z):
    """False discovery rate `fp / (tp + fp)`
    """
    tp, tn, fp, fn = contingency_table(y, z)
    return fp / (tp + fp+pseudocount)


def f1(y, z):
    """F1 score: `2 * (p * r) / (p + r)`, where p=precision and r=recall.
    """
    _recall = recall(y, z)
    _prec = precision(y, z)
    return 2 * (_prec * _recall) / (_prec + _recall+pseudocount)

