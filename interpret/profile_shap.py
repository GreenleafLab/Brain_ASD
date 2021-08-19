#much of this code taken from Alex Tseng, all credit to Alex. 
from .helpers import dinuc_shuffle 
import shap
import tensorflow as tf
import numpy as np

def create_background_counts_chip(model_inputs,bg_size=10,seed=1234):
    input_seq=model_inputs[0]
    cont_counts = model_inputs[1]
    rng = np.random.RandomState(seed)
    input_seq_bg = dinuc_shuffle(input_seq, bg_size, rng=rng)
    cont_counts_bg = np.tile(cont_counts, (bg_size, 1))
    return [input_seq_bg, cont_counts_bg]

def create_background_chip(model_inputs, bg_size=10, seed=1234):
    """
    From a pair of single inputs to the model, generates the set of background
    inputs to perform interpretation against.
    Arguments:
        `model_inputs`: a pair of two entries; the first is a single one-hot
            encoded input sequence of shape I x 4; the second is the set of
            control profiles for the model, shaped T x O x 2
        `bg_size`: the number of background examples to generate.
    Returns a pair of arrays as a list, where the first array is G x I x 4, and
    the second array is G x T x O x 2; these are the background inputs. The
    background for the input sequences is randomly dinuceotide-shuffles of the
    original sequence. The background for the control profiles is the same as
    the originals.
    """
    input_seq=model_inputs[0]
    cont_profs = model_inputs[1]
    rng = np.random.RandomState(seed)
    input_seq_bg = dinuc_shuffle(input_seq, bg_size, rng=rng)
    cont_prof_bg = np.tile(cont_profs, (bg_size, 1, 1))
    return [input_seq_bg, cont_prof_bg]

def create_background_chip_1(model_inputs, bg_size=1, seed=1234):
    input_seq=model_inputs[0]
    cont_profs = model_inputs[1]
    rng = np.random.RandomState(seed)
    input_seq_bg = dinuc_shuffle(input_seq, bg_size, rng=rng)
    cont_prof_bg = np.tile(cont_profs, (bg_size, 1, 1))
    return [input_seq_bg, cont_prof_bg]

def create_background_atac(model_inputs, bg_size=10, seed=1234):
    """
    From a pair of single inputs to the model, generates the set of background
    inputs to perform interpretation against.
    Arguments:
        `model_inputs`: a pair of two entries; the first is a single one-hot
            encoded input sequence of shape I x 4; the second is the set of
            control profiles for the model, shaped T x O x 2
        `bg_size`: the number of background examples to generate.
    Returns a pair of arrays as a list, where the first array is G x I x 4, and
    the second array is G x T x O x 2; these are the background inputs. The
    background for the input sequences is randomly dinuceotide-shuffles of the
    original sequence. The background for the control profiles is the same as
    the originals.
    """
    input_seq= model_inputs[0]
    rng = np.random.RandomState(seed)    
    input_seq_bg = dinuc_shuffle(input_seq, bg_size, rng=rng)
    return [input_seq_bg]

def create_background_atac_1(model_inputs, bg_size=1, seed=1234):
    input_seq= model_inputs[0]
    rng = np.random.RandomState(seed)    
    input_seq_bg = dinuc_shuffle(input_seq, bg_size, rng=rng)
    return [input_seq_bg]


def combine_mult_and_diffref_chip(mult, orig_inp, bg_data):
    """
    Computes the hypothetical contribution of any base along the input sequence
    to the final output, given the multipliers for the input sequence
    background. This will simulate all possible base identities as compute a
    "difference-from-reference" for each possible base, averaging the product
    of the multipliers with the differences, over the base identities. For the
    control profiles, the returned contribution is 0.
    Arguments:
        `mult`: multipliers for the background data; a pair of a G x I x 4 array
            and a G x T x O x 2 array
        `orig_inp`: the target inputs to compute contributions for; a pair of an
            I x 4 array and a T x O x 2 array
        `bg_data`: the background data; a pair of a G x I x 4 array and a
            G x T x O x 2 array
    Returns a pair of importance scores as a list: an I x 4 array and a
    T x O x 2 zero-array.
    This function is necessary for this specific implementation of DeepSHAP. In
    the original DeepSHAP, the final step is to take the difference of the input
    sequence to each background sequence, and weight this difference by the
    contribution multipliers for the background sequence. However, all
    differences to the background would be only for the given input sequence
    (i.e. the actual importance scores). To get the hypothetical importance
    scores efficiently, we try every possible base for the input sequence, and
    for each one, compute the difference-from-reference and weight by the
    multipliers separately. This allows us to compute the hypothetical scores
    in just one pass, instead of running DeepSHAP many times. To get the actual
    scores for the original input, simply extract the entries for the bases in
    the real input sequence.
    """
    # Reassign arguments to better names; this specific implementation of
    # DeepSHAP requires the arguments to have the above names
    input_seq_bg_mults, cont_profs_bg_mults= mult
    input_seq, cont_profs= orig_inp
    input_seq_bg, cont_profs_bg = bg_data
    # Allocate array to store hypothetical scores, one set for each background
    # reference (i.e. each difference-from-reference)
    input_seq_hyp_scores_eachdiff = np.empty_like(input_seq_bg,dtype='float64')
    # Loop over the 4 input bases
    for i in range(input_seq.shape[-1]):
        # Create hypothetical input of all one type of base
        hyp_input_seq = np.zeros_like(input_seq)
        hyp_input_seq[:, i] = 1

        # Compute difference from reference for each reference
        diff_from_ref = np.expand_dims(hyp_input_seq, axis=0) - input_seq_bg
        # Shape: G x I x 4

        # Weight difference-from-reference by multipliers
        contrib = diff_from_ref * input_seq_bg_mults

        # Sum across bases axis; this computes the actual importance score AS IF
        # the target sequence were all that base
        input_seq_hyp_scores_eachdiff[:, :, i] = np.sum(contrib, axis=-1)
    # Average hypothetical scores across background
    # references/diff-from-references
    input_seq_hyp_scores = np.mean(input_seq_hyp_scores_eachdiff, axis=0)
    cont_profs_hyp_scores = np.zeros_like(cont_profs)  # All 0s
    return [input_seq_hyp_scores,cont_profs_hyp_scores]
    
def combine_mult_and_diffref_atac(mult, orig_inp, bg_data):
    """
    Computes the hypothetical contribution of any base along the input sequence
    to the final output, given the multipliers for the input sequence
    background. This will simulate all possible base identities as compute a
    "difference-from-reference" for each possible base, averaging the product
    of the multipliers with the differences, over the base identities. For the
    control profiles, the returned contribution is 0.
    Arguments:
        `mult`: multipliers for the background data; a pair of a G x I x 4 array
            and a G x T x O x 2 array
        `orig_inp`: the target inputs to compute contributions for; a pair of an
            I x 4 array and a T x O x 2 array
        `bg_data`: the background data; a pair of a G x I x 4 array and a
            G x T x O x 2 array
    Returns a pair of importance scores as a list: an I x 4 array and a
    T x O x 2 zero-array.
    This function is necessary for this specific implementation of DeepSHAP. In
    the original DeepSHAP, the final step is to take the difference of the input
    sequence to each background sequence, and weight this difference by the
    contribution multipliers for the background sequence. However, all
    differences to the background would be only for the given input sequence
    (i.e. the actual importance scores). To get the hypothetical importance
    scores efficiently, we try every possible base for the input sequence, and
    for each one, compute the difference-from-reference and weight by the
    multipliers separately. This allows us to compute the hypothetical scores
    in just one pass, instead of running DeepSHAP many times. To get the actual
    scores for the original input, simply extract the entries for the bases in
    the real input sequence.
    """
    # Reassign arguments to better names; this specific implementation of
    # DeepSHAP requires the arguments to have the above names
    input_seq_bg_mults = mult[0]
    input_seq = orig_inp[0]
    input_seq_bg = bg_data[0]
    # Allocate array to store hypothetical scores, one set for each background
    # reference (i.e. each difference-from-reference)
    input_seq_hyp_scores_eachdiff = np.empty_like(input_seq_bg,dtype='float64')
    
    # Loop over the 4 input bases
    for i in range(input_seq.shape[-1]):
        # Create hypothetical input of all one type of base
        hyp_input_seq = np.zeros_like(input_seq)
        hyp_input_seq[:, i] = 1

        # Compute difference from reference for each reference
        diff_from_ref = np.expand_dims(hyp_input_seq, axis=0) - input_seq_bg
        # Shape: G x I x 4

        # Weight difference-from-reference by multipliers
        contrib = diff_from_ref * input_seq_bg_mults

        # Sum across bases axis; this computes the actual importance score AS IF
        # the target sequence were all that base
        input_seq_hyp_scores_eachdiff[:, :, i] = np.sum(contrib, axis=-1)

    # Average hypothetical scores across background
    # references/diff-from-references
    input_seq_hyp_scores = np.mean(input_seq_hyp_scores_eachdiff, axis=0)
    return [input_seq_hyp_scores]


def create_explainer(model, ischip, task_index=None,bg_size=10):
    """
    Given a trained Keras model, creates a Shap DeepExplainer that returns
    hypothetical scores for the input sequence.
    Arguments:
        `model`: a model from `profile_model.profile_tf_binding_predictor`
        `task_index`: a specific task index (0-indexed) to perform explanations
            from (i.e. explanations will only be from the specified outputs); by
            default explains all tasks
    Returns a function that takes in input sequences and control profiles, and
    outputs hypothetical scores for the input sequences.
    """
    prof_output = model.output[0]  # Shape: B x T x O x 2 (logits)
    
    # As a slight optimization, instead of explaining the logits, explain
    # the logits weighted by the probabilities after passing through the
    # softmax; this exponentially increases the weight for high-probability
    # positions, and exponentially reduces the weight for low-probability
    # positions, resulting in a more cleaner signal

    # First, center/mean-normalize the logits so the contributions are
    # normalized, as a softmax would do
    logits = prof_output - \
        tf.reduce_mean(prof_output, axis=1, keepdims=True)

    # Stop gradients flowing to softmax, to avoid explaining those
    logits_stopgrad = tf.stop_gradient(logits)
    probs = tf.nn.softmax(logits_stopgrad, axis=1)

    logits_weighted = logits * probs  # Shape: B x T x O x 2
    if task_index is not None:
        logits_weighted = logits_weighted[:,:, task_index : task_index + 1]
    prof_sum = tf.reduce_sum(logits_weighted, axis=(1, 2))
    if ischip==True:
        if bg_size==10:
            create_background=create_background_chip
        elif bg_size==1:
            create_background=create_background_chip_1
        combine_mult_and_diffref=combine_mult_and_diffref_chip
        model_input=[model.input[0],model.input[1]]
    else:
        if bg_size==10:
            create_background=create_background_atac
        elif bg_size==1:
            create_background=create_background_atac_1
        combine_mult_and_diffref=combine_mult_and_diffref_atac
        model_input=model.input

    explainer = shap.DeepExplainer(
        model=(model_input, prof_sum),
        data=create_background,
        combine_mult_and_diffref=combine_mult_and_diffref
    )

    def explain_fn(input_seqs,control_profile):
        """
        Given input sequences and control profiles, returns hypothetical scores
        for the input sequences.
        Arguments:
            `input_seqs`: a B x I x 4 array
            `cont_profs`: a B x T x O x 4 array
        Returns a B x I x 4 array containing hypothetical importance scores for
        each of the B input sequences.
        """
        if control_profile is not None: 
            return explainer.shap_values([input_seqs,control_profile], progress_message=None)
        else:
            return explainer.shap_values([input_seqs], progress_message=None)

    return explain_fn

