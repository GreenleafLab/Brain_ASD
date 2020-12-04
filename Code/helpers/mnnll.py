import numpy as np
import scipy

def multinomial_log_probs(category_log_probs, trials, query_counts):
    """
    Defines multinomial distributions and computes the probability of seeing
    the queried counts under these distributions. This defines D different
    distributions (that all have the same number of classes), and returns D
    probabilities corresponding to each distribution.
    Arguments:
        `category_log_probs`: a D x N array containing log probabilities (base
            e) of seeing each of the N classes/categories
        `trials`: a D-array containing the total number of trials for each
            distribution (can be different numbers)
        `query_counts`: a D x N array containing the observed count of each
            category in each distribution; the probability is computed for these
            observations
    Returns a D-array containing the log probabilities (base e) of each observed
    query with its corresponding distribution. Note that D can be replaced with
    any shape (i.e. only the last dimension is reduced).
    """
    # Multinomial probability = n! / (x1!...xk!) * p1^x1 * ... pk^xk
    # Log prob = log(n!) - (log(x1!) ... + log(xk!)) + x1log(p1) ... + xklog(pk)
    log_n_fact = scipy.special.gammaln(trials + 1)
    log_counts_fact = scipy.special.gammaln(query_counts + 1)
    log_counts_fact_sum = np.sum(log_counts_fact, axis=-1)
    log_prob_pows = category_log_probs * query_counts  # Elementwise
    log_prob_pows_sum = np.sum(log_prob_pows, axis=-1)
    return log_n_fact - log_counts_fact_sum + log_prob_pows_sum

def profile_multinomial_nll(
    true_profs, log_pred_profs, true_counts, batch_size=200
):
    """
    Computes the negative log likelihood of seeing the true profile, given the
    probabilities specified by the predicted profile. The NLL is computed
    separately for each sample, task, and strand, but the results are averaged
    across the strands.
    Arguments:
        `true_profs`: N x T x O x 2 array, where N is the number of
            examples, T is the number of tasks, and O is the output profile
            length; contains the true profiles for each for each task and
            strand, as RAW counts
        `log_pred_profs`: a N x T x O x 2 array, containing the predicted
            profiles for each task and strand, as LOG probabilities
        `true_counts`: a N x T x 2 array, containing the true total counts
            for each task and strand
        `batch_size`: performs computation in a batch size of this many samples
    Returns an N x T array, containing the strand-pooled multinomial NLL for
    each sample and task.
    """
    num_samples = true_profs.shape[0]
    num_tasks = true_profs.shape[1]
    nlls = np.empty((num_samples, num_tasks))
    for start in range(0, num_samples, batch_size):
        end = start + batch_size
        true_profs_batch = true_profs[start:end]
        log_pred_profs_batch = log_pred_profs[start:end]
        true_counts_batch = true_counts[start:end]
        # Swap axes on profiles to make them B x T x 2 x O
        true_profs_batch = np.swapaxes(true_profs_batch, 2, 3)
        log_pred_profs_batch = np.swapaxes(log_pred_profs_batch, 2, 3)
        nll_batch = -multinomial_log_probs(
            log_pred_profs_batch, true_counts_batch, true_profs_batch
        )
        nll_batch_mean = np.mean(nll_batch, axis=2)  # Shape: B x T
        nlls[start:end] = nll_batch_mean
    return nlls
