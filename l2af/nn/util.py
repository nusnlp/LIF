import torch
import torch.nn.functional as F
from allennlp.nn.util import masked_softmax
from allennlp.nn.util import get_text_field_mask


def sequential_weighted_avg(x, weights):
    """Return a sequence by weighted averaging of x (a sequence of vectors).
    Args:
        x: batch * len2 * hdim
        weights: batch * len1 * len2, sum(dim = 2) = 1
    Output:
        x_avg: batch * len1 * hdim
    """
    # return weights.unsqueeze(1).bmm(x).squeeze(1)
    return weights.bmm(x)
