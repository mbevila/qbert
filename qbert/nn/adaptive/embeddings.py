# https://github.com/AranKomat/adapinp

import torch.nn as nn
from torch.nn import functional as F


class AdaptiveInput(nn.Module):
    r"""
    * :attr:`cutoffs` should be an ordered Sequence of integers sorted
      in the increasing order.
      It controls number of clusters and the partitioning of targets into
      clusters. For example setting ``cutoffs = [10, 100, 1000]``
      means that first `10` targets will be assigned
      to the 'head' of the adaptive input, targets `11, 12, ..., 100` will be
      assigned to the first cluster, and targets `101, 102, ..., 1000` will be
      assigned to the second cluster, while targets
      `1001, 1002, ..., n_classes - 1` will be assigned
      to the last, third cluster
    * :attr:`div_value` is used to compute the size of each additional cluster,
      which is given as
      :math:`\left\lfloor\frac{in\_features}{div\_value^{idx}}\right\rfloor`,
      where :math:`idx` is the cluster index (with clusters
      for less frequent words having larger indices,
      and indices starting from :math:`1`).
    .. warning::
        Labels passed as inputs to this module should be sorted accoridng to
        their frequency. This means that the most frequent label should be
        represented by the index `0`, and the least frequent
        label should be represented by the index `n_classes - 1`.
    Args:
        in_features (int): Number of features in the input tensor
        n_classes (int): Number of classes in the dataset.
        cutoffs (Sequence): Cutoffs used to assign targets to their buckets.
        div_value (float, optional): value used as an exponent to compute sizes
            of the clusters. Default: 4.0
    Returns:
        ``NamedTuple`` with ``output`` and ``loss`` fields:
            * **output** is a Tensor of size ``N`` containing computed target
              log probabilities for each example
            * **loss** is a Scalar representing the computed negative
              log likelihood loss
    Shape:
        - input: :math:`(N)` where each value satisfies :math:`0 <= target[i] <= n\_classes`
        - output: :math:`(N, in\_features)`
    This implementation and the above description are heavily cited from the softmax counterpart from
    https://pytorch.org/docs/stable/_modules/torch/nn/nn/adaptive.html
    """

    def __init__(self, n_classes, in_features, cutoffs=None,
                 div_value=4., head_bias=False):
        super(AdaptiveInput, self).__init__()
        if not cutoffs:
            cutoffs = [10000, 60000, 190000]
        cutoffs = list(cutoffs)

        if (cutoffs != sorted(cutoffs)) \
                or (min(cutoffs) <= 0) \
                or (max(cutoffs) >= (n_classes - 1)) \
                or (len(set(cutoffs)) != len(cutoffs)) \
                or any([int(c) != c for c in cutoffs]):
            raise ValueError("cutoffs should be a sequence of unique, positive "
                             "integers sorted in an increasing order, where "
                             "each value is between 1 and n_classes-1")

        self.in_features = in_features
        self.n_classes = n_classes
        self.cutoffs = cutoffs + [n_classes]
        self.div_value = div_value
        self.head_bias = head_bias

        self.n_clusters = len(self.cutoffs) - 1
        self.head_size = self.cutoffs[0]

        self.head = nn.Sequential(nn.Embedding(self.head_size, self.in_features),
                                  nn.Linear(self.in_features, self.in_features, bias=self.head_bias))

        self.tail_embs = nn.ModuleList()
        self.tail_lins = nn.ModuleList()

        for i in range(self.n_clusters):
            hsz = int(self.in_features // (self.div_value ** (i + 1)))
            osz = self.cutoffs[i + 1] - self.cutoffs[i]

            self.tail_embs.append(nn.Embedding(osz, hsz))
            self.tail_lins.append(nn.Embedding(hsz, self.in_features))

    def forward(self, input):

        if len(input.shape) > 1:
            shape = input.shape
            input = input.contiguous().view(-1)
        else:
            shape = None

        used_rows = 0
        input_size = list(input.size())

        output = input.new_zeros(input_size + [self.in_features]).float()

        cutoff_values = [0] + self.cutoffs
        for i in range(len(cutoff_values) - 1):

            low_idx = cutoff_values[i]
            high_idx = cutoff_values[i + 1]

            input_mask = (input >= low_idx) & (input < high_idx)
            row_indices = input_mask.nonzero().squeeze()

            if row_indices.numel() == 0:
                continue
            idx = input[input_mask] - low_idx
            if i == 0:
                out = self.head(idx)
            else:
                out = self.tail_embs[i-1](idx)
                out = F.linear(out, self.tail_lins[i-1].weight.t())
            output.index_copy_(0, row_indices, out)
            used_rows += row_indices.numel()

        if used_rows != input_size[0]:
            raise RuntimeError("Target values should be in [0, {}], "
                               "but values in range [{}, {}] "
                               "were found. ".format(self.n_classes - 1,
                                                     input.min().item(),
                                                     input.max().item()))

        if shape is not None:
            output = output.view(*shape, -1)
            pass
        return output

    def make_tied_decoder(self):
        decoder = nn.AdaptiveLogSoftmaxWithLoss(
            self.in_features, self.n_classes, self.cutoffs[:-1], self.div_value, head_bias=self.head_bias,
        )
        # self.head[0].weight.data = decoder.head.weight[:self.head_size].data
        self.head[0].weight.data = decoder.head.weight[:self.head_size].data

        for cluster_in1, cluster_in2, cluster_out in zip(self.tail_embs, self.tail_lins, decoder.tail):
            cluster_in1.weight = cluster_out[1].weight
            cluster_in2.weight = cluster_out[0].weight
        print()
        return decoder


if __name__ == "__main__":
    import torch

    x = torch.randint(0, 100, (10, 32)).long()
    inp = AdaptiveInput(128, 100, cutoffs=[4, 8, 16])
    out = inp.make_tied_decoder()
    x = inp(x)


