import torch

from qbert.nn import LockedDropout
from qbert.nn.attentive.submodules import DirectionalTransformerBlock, TransformerBlock

DEFAULT_HPARAMS = {
    "nlayers": 6,
    "hid_features": None,
    "num_heads": 4,
    "attention_dropout": 0.3,
    "dropout": 0.1,
    "dropouth": 0.1,
    "relu_dropout": 0.3,
    "encoder_normalize_before": False,
    "encoder_ffn_embed_dim": None,
}

class TransformerLayer(torch.nn.Module):

    DEFAULT_HPARAMS = DEFAULT_HPARAMS.copy()

    def __init__(self, in_features, add_linear_in=False, hparams=None, **kwargs):
        super().__init__()

        defaults = self.DEFAULT_HPARAMS.copy()
        if hparams is not None:
            defaults.update(hparams)
        if kwargs:
            defaults.update(kwargs)
        self.hparams = hparams = defaults
        self.hparams["hid_features"] = self.hparams["hid_features"] or in_features

        self.in_features = in_features
        self.hid_features = self.hparams["hid_features"]
        self.out_features = self.hid_features
        self.use_fc_in = add_linear_in
        if not add_linear_in:
            assert self.hid_features == self.in_features, "if use_fc_in == False, hid_features must be None or == in_features"

        self.noutputs = hparams["nlayers"]

        if self.use_fc_in:
            self.linear_in = torch.nn.Linear(self.in_features, self.hid_features)
        else:
            self.linear_in = None
            assert self.in_features == self.hid_features

        self._alpha = 0
        self._beta = 0

        self.dropout = LockedDropout(hparams["dropout"])
        self.dropouth = LockedDropout(hparams["dropouth"])

        def make_tf():
            return TransformerBlock(
                in_features=self.hid_features,
                num_heads=hparams["num_heads"],
                hparams=hparams,
            )
        self.tf_blocks = torch.nn.ModuleList(
                [make_tf() for _ in range(hparams["nlayers"])])

    def forward(self, input, padding_mask=None, all_states=False):
        outs = []
        self.attention_weights = []

        out = input
        if self.linear_in:
            out = self.linear_in(out)

        for i, block in enumerate(self.tf_blocks):
            out, weights = block(out, encoder_padding_mask=padding_mask)
            if i < (len(self.tf_blocks) - 1):
                out = self.dropouth(out)
            else:
                out = self.dropout(out)
            outs.append(out)
            self.attention_weights.append(weights)

        out = outs[-1]

        self._alpha = sum([o.pow(2).mean() for o in outs])

        if all_states:
            outs = torch.stack([outs], 3)
            return out, outs
        else:
            return out

    def alpha(self):
        return self._alpha

    def beta(self):
        return self._beta


class TimeShiftedTransformerLayer(TransformerLayer):

    DEFAULT_HPARAMS = DEFAULT_HPARAMS.copy()

    def __init__(self, in_features, bidirectional=True, backward=False, mode="add", add_linear_in=False, hparams=None, invert_last=False, **kwargs):
        super().__init__(
            in_features=in_features,
            add_linear_in=add_linear_in,
            hparams=hparams,
            **kwargs,
        )

        if invert_last:
            assert bidirectional

        self.backward = backward
        self.bidirectional = bidirectional
        self.mode = "add"
        self.invert_last = invert_last
        self.out_features = self.hid_features * 2 if self.bidirectional and self.mode == "cat" else self.hid_features
        self.noutputs = (self.hparams["nlayers"] + self.invert_last, self.hparams["nlayers"] + self.invert_last)

        def make_tf():
            return DirectionalTransformerBlock(
                in_features=self.hid_features,
                num_heads=self.hparams["num_heads"],
                hparams=self.hparams,
            )
        self.tf_blocks = torch.nn.ModuleList(
            [make_tf() for _ in range(self.hparams["nlayers"])])

        if self.backward and (not self.bidirectional):
            self.tf_blocks_bw = self.tf_blocks
            del self.tf_blocks

        if self.bidirectional:
            def make_tf():
                return DirectionalTransformerBlock(
                    in_features=self.hid_features,
                    num_heads=self.hparams["num_heads"],
                    hparams=self.hparams,
                )
            self.tf_blocks_bw = torch.nn.ModuleList(
                [make_tf() for _ in range(self.hparams["nlayers"])])
        if self.bidirectional and self.invert_last:
            self.transformer_bw_on_fw = make_tf()
            self.transformer_fw_on_bw = make_tf()
        self.register_buffer("_t", torch.tensor([]))

    def _forward_unidirectional(self, input, padding_mask, backward=False):
        outs = []
        if not backward:
            self.last_weights_fw = []
            out = input
            for i, block in enumerate(self.tf_blocks):
                out, weights = block(out, encoder_padding_mask=padding_mask)
                if i < (len(self.tf_blocks) - 1):
                    out = self.dropouth(out)
                else:
                    out = self.dropout(out)
                outs.append(out)
                self.last_weights_fw.append(weights)
        else:
            self.last_weights_bw = []
            out = torch.flip(input, dims=[0])
            for i, block in enumerate(self.tf_blocks_bw):
                out, weights = block(out, encoder_padding_mask=padding_mask)
                if i < (len(self.tf_blocks) - 1):
                    out = self.dropouth(out)
                else:
                    out = self.dropout(out)
                outs.append(torch.flip(out, dims=[0]))
                self.last_weights_bw.append(weights)
        self._alpha += sum([o.pow(2).mean() for o in outs])
        return outs

    def forward(self, input, padding_mask=None, all_states=False):
        self._alpha = 0.

        if self.use_fc_in:
            input = self.linear_in(input)

        if self.bidirectional:
            pad = self._t.new_zeros(1, input.size(1), input.size(2))

            outs_fw = self._forward_unidirectional(
                torch.cat([pad, input[:-1]], dim=0),
                padding_mask,
                backward=False)
            outs_bw = self._forward_unidirectional(
                torch.cat([input[1:], pad], dim=0),
                padding_mask,
                backward=True)

            if self.mode == "cat":
                out = torch.cat([outs_fw[-1], outs_bw[-1]], dim=-1)
            else:
                out = outs_fw[-1] + outs_bw[-1]
            self._alpha /= 2
            if all_states:
                outs_fw = torch.stack([outs_fw], 3)
                outs_bw = torch.stack([outs_bw], 3)
                return out, torch.cat([outs_fw, outs_bw], 2)
            else:
                return out
        else:
            outs = self._forward_unidirectional(
                input,
                padding_mask,
                backward=self.backward)
            out = outs[-1]
            if all_states:
                outs = torch.stack([outs], 3)
                return out, outs
            else:
                return out

    def alpha(self):
        return self._alpha

    def beta(self):
        return self._beta

