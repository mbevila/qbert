import torch

from qbert.nn.recurrent.core import BiRNNStack, RNNStack

class RecurrentLayer(torch.nn.Module):

    DEFAULT_HPARAMS = {
        "nlayers": 3,
        "hid_features": None,
        "rnn_type": "LSTM",
        "dropout": 0.5,
        "dropouth": 0.5,
        "wdrop": 0.65,
        "layer_norm": False,
    }

    def __init__(self, in_features, out_features, bidirectional=False, hparams=None, **kwargs):
        super().__init__()

        defaults = self.DEFAULT_HPARAMS.copy()
        if hparams is not None:
            defaults.update(hparams)
        if kwargs:
            defaults.update(kwargs)
        self.hparams = hparams = defaults
        if not hparams["hid_features"]:
            hparams["hid_features"] = out_features

        self.in_features = in_features
        self.out_features = out_features * 2 if bidirectional else out_features
        self.bidirectional = bidirectional
        self.hparams = hparams

        if bidirectional:
            self.fw = BiRNNStack(in_features=in_features, out_features=out_features, **hparams)
        else:
            self.fw = RNNStack(in_features=in_features, out_features=out_features, **hparams)

    def forward(self, input, hidden=None, all_states=False):
        out, hidden, rhs, drhs = self.fw(input, hidden, return_h=True)
        self._alpha = sum(dropped_rnn_h.pow(2).mean() for dropped_rnn_h in drhs[-1:])
        self._beta = sum((rnn_h[1:] - rnn_h[:-1]).pow(2).mean() for rnn_h in rhs[-1:])

        if all_states:
            return out, hidden, torch.stack(rhs, dim=3)
        else:
            return out, hidden

    def init_hidden(self, bsz):
        if self.bidirectional:
            return (self.fw.init_hidden(bsz), self.bw.init_hidden(bsz))
        return self.fw.init_hidden(bsz)

    def alpha(self):
        return self._alpha

    def beta(self):
        return self._beta


class TimeShiftedRecurrentLayer(torch.nn.Module):
    DEFAULT_HPARAMS = {
        "nlayers": 3,
        "hid_features": None,
        "rnn_type": "LSTM",
        "dropout": 0.4,
        "dropouth": 0.3,
        "wdrop": 0.,
        "layer_norm": False,
    }

    def __init__(self, in_features, out_features, bidirectional=False, hparams=None, **kwargs):
        super().__init__()

        defaults = self.DEFAULT_HPARAMS.copy()
        if hparams is not None:
            defaults.update(hparams)
        if kwargs:
            defaults.update(kwargs)
        self.hparams = hparams = defaults
        if not hparams["hid_features"]:
            hparams["hid_features"] = out_features

        self.in_features = in_features
        self.out_features = out_features * 2 if bidirectional else out_features
        self.bidirectional = bidirectional

        self.fw = RNNStack(in_features=in_features, out_features=out_features, **hparams)
        self.fw.init_weights()
        if bidirectional:
            self.bw = RNNStack(in_features=in_features, out_features=out_features, **hparams)
            self.bw.init_weights()
        self.register_buffer("_t", torch.tensor(0).float())


    def forward(self, input, hidden=None, all_states=False):
        if hidden is None:
            hidden = self.init_hidden(input.size(1))
        if self.bidirectional:
            fw_hid, bw_hid = hidden
        else:
            fw_hid = hidden

        if self.bidirectional:

            bd_pad = self._t.new_zeros(1, input.size(1), input.size(2))
            fw_inp = torch.cat([bd_pad, input], 0)
            fw_out, fw_hid, rhs_fw, drhs = self.fw(fw_inp, fw_hid, return_h=True)
            self._alpha = sum(dropped_rnn_h.pow(2).mean() for dropped_rnn_h in drhs[-1:])
            self._beta = sum((rnn_h[1:] - rnn_h[:-1]).pow(2).mean() for rnn_h in rhs_fw[-1:])

            # bw_inp = torch.cat([bd_pad, torch.flip(input[1:], [0])], 0)
            bw_inp = torch.cat([bd_pad, torch.flip(input, [0])], 0)
            bw_out, bw_hid, rhs_bw, drhs = self.bw(bw_inp, bw_hid, return_h=True)
            self._alpha += sum(dropped_rnn_h.pow(2).mean() for dropped_rnn_h in drhs[-1:])
            self._beta += sum((rnn_h[1:] - rnn_h[:-1]).pow(2).mean() for rnn_h in rhs_bw[-1:])
            bw_out = torch.flip(bw_out, [0])

            if self.aligned:
                out = torch.cat([fw_out[1:], bw_out[:-1]], 2)  # removes shifting padding
            else:
                out = torch.cat([fw_out[:-1], bw_out[1:]], 2)  # removes last timestep in fw and first in bw
            hidden = (fw_hid, bw_hid)

            if all_states:
                all_states = torch.cat([
                    torch.stack(rhs_fw, dim=3)[1:],
                    torch.flip(torch.stack(rhs_bw, dim=3), [0])[:-1]],
                    dim=2,
                )
                return out, hidden, all_states
            else:
                return out, hidden
        else:
            fw_out, fw_hid, rhs_fw, drhs = self.fw(input, fw_hid, return_h=True)
            self._alpha = sum(dropped_rnn_h.pow(2).mean() for dropped_rnn_h in drhs[-1:])
            self._beta = sum((rnn_h[1:] - rnn_h[:-1]).pow(2).mean() for rnn_h in rhs_fw[-1:])
            if all_states:
                all_states = torch.stack(rhs_fw, dim=3)[1:]
                return fw_out, hidden, all_states
            else:
                return fw_out, hidden

    def init_hidden(self, bsz):
        if self.bidirectional:
            return (self.fw.init_hidden(bsz), self.bw.init_hidden(bsz))
        return self.fw.init_hidden(bsz)

    def alpha(self):
        return self._alpha

    def beta(self):
        return self._beta

    def reset(self):
        if self.hparams.rnn_type == 'QRNN':
            [r.reset() for r in self.fw.rnns]
            if self.bidirectional:
                [r.reset() for r in self.bw.rnns]