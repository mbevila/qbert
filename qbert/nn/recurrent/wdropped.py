import torch

def weight_drop_apply(module, weights, dropout):
    regular_weights = []
    for name_w in weights:
        regular_w = getattr(module, name_w)
        dropped_w = torch.nn.functional.dropout(regular_w, p=dropout, training=module.training)
        module.register_parameter(name_w,
                                  torch.nn.Parameter(dropped_w, requires_grad=False)
                                  )
        regular_weights.append(regular_w)
    return regular_weights

def weight_drop_undo(module, weights, regular_weights):
    for name_w, regular_w in zip(weights, regular_weights):
        module.register_parameter(name_w, regular_w)

class WeightDropLSTM(torch.nn.LSTM):
    """
    Wrapper around :class:`torch.nn.LSTM` that adds ``weight_dropout`` named argument.

    Args:
        weight_dropout (float): The probability a weight will be dropped.
    """

    def __init__(self, *args, weight_dropout=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_dropout = weight_dropout
        self._weight_dropped_weights = ['weight_hh_l' + str(i) for i in range(self.num_layers)]

    def forward(self, input, hx=None):
        if self.training and self.weight_dropout:
            regular_weights = weight_drop_apply(self, self._weight_dropped_weights, self.weight_dropout)
            out = super().forward(input=input, hx=hx)
            weight_drop_undo(self, self._weight_dropped_weights, regular_weights)
        else:
            out = super().forward(input=input, hx=hx)
        return out

class WeightDropGRU(torch.nn.GRU):
    """
    Wrapper around :class:`torch.nn.GRU` that adds ``weight_dropout`` named argument.

    Args:
        weight_dropout (float): The probability a weight will be dropped.
    """

    def __init__(self, *args, weight_dropout=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_dropout = weight_dropout
        self._weight_dropped_weights = ['weight_hh_l' + str(i) for i in range(self.num_layers)]

    def forward(self, input, hx=None):
        if self.training and self.weight_dropout:
            regular_weights = weight_drop_apply(self, self._weight_dropped_weights, self.weight_dropout)
            out = super().forward(input=input, hx=hx)
            weight_drop_undo(self, self._weight_dropped_weights, regular_weights)
        else:
            out = super().forward(input=input, hx=hx)
        return out