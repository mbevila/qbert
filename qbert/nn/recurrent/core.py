import torch

from qbert.nn import LockedDropout, WeightDrop
from .wdropped import WeightDropLSTM, WeightDropGRU

class RNNStack(torch.nn.Module):

    def __init__(
            self,
            in_features,
            hid_features,
            out_features,
            nlayers,
            rnn_type="LSTM",
            dropout=0.5,
            dropouth=0.5,
            wdrop=0.,
            layer_norm=False,
    ):
        super().__init__()

        self.in_features = in_features
        self.hid_features = hid_features
        self.out_features = out_features
        self.nlayers = nlayers
        self.rnn_type = rnn_type
        self.layer_norm = layer_norm
        self.dropout, self.dropout_layer = dropout, LockedDropout(dropout)
        self.dropouth, self.dropouth_layer = dropouth, LockedDropout(dropouth)
        self.wdrop = wdrop

        assert self.rnn_type in ['LSTM', 'QRNN', 'GRU'], 'RNN type is not supported'

        if self.rnn_type in {'LSTM', 'GRU'}:
            if self.rnn_type == 'LSTM':
                RNN = torch.nn.LSTM
                #RNN = WeightDropLSTM
            else:
                RNN = torch.nn.GRU
                #RNN = WeightDropGRU
            self.rnns = [RNN(
                input_size=self.in_features if l == 0 else self.hid_features,
                hidden_size=self.hid_features if l < self.nlayers - 1 else self.out_features,
                num_layers=1,
                dropout=0,
                #weight_dropout=self.wdrop,
                )
                for l in range(self.nlayers)]
            if self.wdrop:
                self.rnns = [WeightDrop(r, ["weight_hh_l0"], dropout=self.wdrop) for r in self.rnns]
        elif self.rnn_type == 'QRNN':
            raise NotImplementedError
            # from torchqrnn import QRNNLayer
            # self.rnns = [QRNNLayer(
            #     input_size=self.insize if l == 0 else self.hidsize,
            #     hidden_size=self.hidsize if l < self.nlayers - 1 else self.outsize,
            #     save_prev_x=True, zoneout=0, window=2 if l == 0 else 1, output_gate=True)
            #     for l in range(self.nlayers)]
            # if self.wdrop:
            #     for rnn in self.rnns:
            #         rnn.linear = WeightDrop(rnn.linear, ['weight'], dropout=self.wdrop)

        if self.layer_norm:
            self.layer_norms = torch.nn.ModuleList([torch.nn.LayerNorm(r.hidden_size) for r in self.rnns])
        self.rnns = torch.nn.ModuleList(self.rnns)


    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return [(
                weight.new(1, bsz, self.hid_features if l != self.nlayers - 1 else self.out_features).zero_(),
                weight.new(1, bsz, self.hid_features if l != self.nlayers - 1 else self.out_features).zero_())
                for l in range(self.nlayers)]
        elif self.rnn_type == 'QRNN' or self.rnn_type == 'GRU':
            return [
                weight.new(1, bsz, self.hid_features if l != self.nlayers - 1 else self.out_features).zero_()
                for l in range(self.nlayers)]

    def init_weights(self):
        pass

    def forward(self, input, hidden=None, return_h=False):
        if hidden is None:
            hidden = self.init_hidden(input.size(1))
        raw_output = input
        new_hidden = []
        raw_outputs = []
        outputs = []
        for l, rnn in enumerate(self.rnns):
            raw_output, new_h = rnn(raw_output, hidden[l])
            new_hidden.append(new_h)
            raw_outputs.append(raw_output)
            if self.layer_norm:
                raw_output = self.layer_norms[l](raw_output)
            if l != self.nlayers - 1:
                raw_output = self.dropouth_layer(raw_output)
                outputs.append(raw_output)
        hidden = new_hidden
        output = self.dropout_layer(raw_output)
        outputs.append(output)
        if return_h:
            return output, hidden, raw_outputs, outputs
        return output, hidden


class BiRNNStack(torch.nn.Module):

    def __init__(
            self,
            in_features,
            hid_features,
            out_features,
            nlayers,
            rnn_type="LSTM",
            dropout=0.5,
            dropouth=0.5,
            wdrop=0.,
            layer_norm=False,
    ):

        super(BiRNNStack, self).__init__()

        self.in_features = in_features
        self.hid_features = hid_features
        self.out_features = out_features * 2
        self.nlayers = nlayers
        self.rnn_type = rnn_type
        self.dropout = dropout
        self.dropouth = dropouth
        self.wdrop = wdrop
        self.layer_norm = layer_norm

        self.init_module()

    def init_module(self):

        self.dropout_layer = LockedDropout(self.dropout)
        self.dropouth_layer = LockedDropout(self.dropouth)

        insize = self.in_features
        hidsize_out = self.hid_features
        hidsize_in = self.hid_features * 2
        outsize = self.out_features // 2

        assert self.rnn_type in ['LSTM', 'QRNN', 'GRU'], 'RNN type is not supported'

        if self.rnn_type in {'LSTM', 'GRU'}:
            if self.rnn_type == 'LSTM':
                RNN = WeightDropLSTM
            else:
                RNN = WeightDropGRU

            def make_rnns():
                rnns = [RNN(
                    input_size=insize if l == 0 else hidsize_in,
                    hidden_size=hidsize_out if l < self.nlayers - 1 else outsize,
                    num_layers=1,
                    dropout=0,
                    weight_dropout=self.wdrop
                )
                    for l in range(self.nlayers)]
                return torch.nn.ModuleList(rnns)

            self.rnns_fw = make_rnns()
            self.rnns_bw = make_rnns()
            if self.layer_norm:
                self.layer_norms = [torch.nn.LayerNorm(f.hidden_size + b.hidden_size) \
                                    for f, b in zip(self.rnns_fw, self.rnns_bw)]
                self.layer_norms = torch.nn.ModuleList(self.layer_norms)
            else:
                self.layer_norms = None
        elif self.rnn_type == 'QRNN':
            raise NotImplementedError
            # from torchqrnn import QRNNLayer
            # def make_qrnns():
            #     rnns = [QRNNLayer(
            #         input_size=insize if l == 0 else hidsize_in,
            #         hidden_size=hidsize_out if l < self.nlayers - 1 else outsize,
            #         save_prev_x=True, zoneout=0, window=2 if l == 0 else 1, output_gate=True)
            #         for l in range(self.nlayers)]
            #     if self.wdrop:
            #         for rnn in rnns:
            #             rnn.linear = WeightDrop(rnn.linear, ['weight'], dropout=self.wdrop)
            #     return torch.nn.ModuleList(rnns)
            #
            # self.rnns_fw = make_qrnns()
            # self.rnns_bw = make_qrnns()

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return [(
                weight.new(1, bsz, self.hid_features if l != self.nlayers - 1 else self.out_features).zero_(),
                weight.new(1, bsz, self.hid_features if l != self.nlayers - 1 else self.out_features).zero_())
                for l in range(self.nlayers)]
        elif self.rnn_type == 'QRNN' or self.rnn_type == 'GRU':
            return [
                weight.new(1, bsz, self.hid_features if l != self.nlayers - 1 else self.out_features).zero_()
                for l in range(self.nlayers)]

    def init_weights(self):
        pass

    def forward(self, input, return_h=False):
        raw_output = input
        new_hidden = []
        raw_outputs = []
        outputs = []
        for l, (fw, bw) in enumerate(zip(self.rnns_fw, self.rnns_bw)):
            raw_output1, new_h = fw(raw_output, None)
            raw_output2, _ = bw(torch.flip(raw_output, (0,)), None)
            raw_output = torch.cat([raw_output1, torch.flip(raw_output2, (0,))], -1)
            new_hidden.append(new_h)
            raw_outputs.append(raw_output)
            if self.layer_norms:
                raw_output = self.layer_norms[l](raw_output)
            if l != self.nlayers - 1:
                raw_output = self.dropouth_layer(raw_output)
                outputs.append(raw_output)
        hidden = new_hidden
        output = self.dropout_layer(raw_output)
        outputs.append(output)
        if return_h:
            return output, hidden, raw_outputs, outputs
        return output, hidden


def unroll(tensor):
    shape = tensor.shape[:2]
    return tensor.view(-1, *tensor.shape[2:]), shape


def roll(tensor, shape):
    return tensor.view(*shape, *tensor.shape[1:])
