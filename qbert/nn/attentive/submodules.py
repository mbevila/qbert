import copy

from torch import nn
from torch.nn import functional as F

from qbert.nn.attentive.core import MultiheadAttention, MultiheadAttentionWithDirectionalMask

class TransformerBlock(nn.Module):
    """Encoder layer block.
    In the original paper each operation (multi-head attention or FFN) is
    postprocessed with: `dropout -> add residual -> layernorm`. In the
    tensor2tensor code they suggest that learning is more robust when
    preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.encoder_normalize_before* to ``True``.
    Args:
        args (argparse.Namespace): parsed command-line arguments
    """

    DEFAULT_HPARAMS = {
        "attention_dropout": 0.1,
        "dropout": 0.1,
        "relu_dropout": 0.1,
        "normalize_before": False,
        "ffn_embed_dim": None,
    }

    def __init__(self, in_features, num_heads, hparams={}, **kwargs):
        super().__init__()

        self.hparams = copy.deepcopy(self.DEFAULT_HPARAMS.copy())
        self.hparams.update(hparams)
        hparams = self.hparams
        self.in_features = in_features
        self.num_heads = num_heads

        self.insize = self.outsize = in_features
        self.hidsize = hparams["ffn_embed_dim"] or in_features

        self.self_attn = MultiheadAttention(
            self.in_features, num_heads,
            dropout=hparams["attention_dropout"],
        )
        self.dropout = hparams["dropout"]
        self.relu_dropout = hparams["relu_dropout"]
        self.normalize_before = hparams["normalize_before"]
        self.fc1 = nn.Linear(self.in_features, self.hidsize)
        self.fc2 = nn.Linear(self.hidsize, self.outsize)
        self.layer_norms = nn.ModuleList([nn.LayerNorm(self.outsize) for i in range(2)])

    def forward(self, query, kv=None, encoder_padding_mask=None, attn_mask=None):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.
        Returns:
            encoded output of shape `(batch, src_len, embed_dim)`
        """

        x = query
        if kv is None:
            kv = x

        residual = x
        x = self.maybe_layer_norm(0, x, before=True)
        x, weights = self.self_attn(query=x, key=kv, value=kv,  key_padding_mask=encoder_padding_mask, attn_mask=attn_mask)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(0, x, after=True)

        residual = x
        x = self.maybe_layer_norm(1, x, before=True)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.relu_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(1, x, after=True)
        return x, weights

    def maybe_layer_norm(self, i, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return self.layer_norms[i](x)
        else:
            return x

class DirectionalTransformerBlock(TransformerBlock):

    def __init__(self, in_features, num_heads, mask_backward=False, mask_present=False, hparams={}, **kwargs):
        super().__init__(
            in_features, num_heads, hparams, mask_backward=mask_backward, mask_present=mask_present, **kwargs)
        self.mask_backward = mask_backward
        self.mask_present = mask_present
        self.self_attn = MultiheadAttentionWithDirectionalMask(
            self.in_features, num_heads,
            dropout=self.hparams["attention_dropout"],
            mask_backward=self.mask_backward,
            mask_present=self.mask_present,
        )


if __name__ == "__main__":
    import torch
    from transformer_lm.modules.adaptive.embeddings import AdaptiveInput

    embed_dim = 300
    num_heads = 1
    embedding = AdaptiveInput(1000, embed_dim, cutoffs=[50, 100, 250])

    x = torch.Tensor([
        [12, 15, 1, 24, 36, 7],
        [1, 56, 28, 67, 0, 0],
    ]).long().transpose(1, 0)
    embedded = embedding(x)

    regular_blocks = [TransformerBlock(embed_dim, num_heads).eval() for _ in range(4)]
    out = embedded
    for b in regular_blocks:
        out, weights = b(out)

    print(weights)
    print(weights.sum(2))


    directed_blocks = [DirectionalTransformerBlock(embed_dim, num_heads).eval() for _ in range(4)]
    out, weights = directed_blocks[0](embedded)

    print("First directed layer")
    print(weights)
    print(weights.sum(2))
    print()

    for b in directed_blocks[1:]:
        out, weights = b(out)

    print("Last directed layer")
    print(weights)
    print(weights.sum(1))
    print()

