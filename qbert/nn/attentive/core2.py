# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F

class MultiheadAttention(nn.Module):
    """Multi-headed attention.
    See "Attention Is All You Need" for more details.
    """

    def __init__(
            self,
            in_features,
            num_heads,
            dropout=0.,
            bias=True,
            add_bias_kv=False,
            add_zero_attn=False,
            **kwargs
    ):
        super().__init__()
        self.in_features = in_features
        self.num_heads = num_heads
        self.dropout = dropout
        self.bias = bias
        self.add_bias_kv = add_bias_kv
        self.add_zero_attn = add_zero_attn

        self.head_dim = in_features // num_heads
        assert self.head_dim * num_heads == self.in_features, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.in_proj_weight = Parameter(torch.Tensor(3 * in_features, in_features))
        if bias:
            self.in_proj_bias = Parameter(torch.Tensor(3 * in_features))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = nn.Linear(in_features, in_features, bias=bias)

        if add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, in_features))
            self.bias_v = Parameter(torch.Tensor(1, 1, in_features))
        else:
            self.bias_k = self.bias_v = None

        self.reset_parameters()


    def reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def forward(self, query, key, value, key_padding_mask=None, incremental_state=None,
                need_weights=True, static_kv=False, attn_mask=None):
        """Input shape: Time x Batch x Channel
        Self-attention can be implemented by passing in the same arguments for
        query, key and value. Timesteps can be masked by supplying a T x T mask in the
        `attn_mask` argument. Padding elements can be excluded from
        the key by passing a binary ByteTensor (`key_padding_mask`) with shape:
        batch x src_len, where padding elements are indicated by 1s.
        """

        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.transpose(0, 1)

        qkv_same = query.data_ptr() == key.data_ptr() == value.data_ptr()
        kv_same = key.data_ptr() == value.data_ptr()

        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.in_features
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        assert key.size() == value.size()

        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if 'prev_key' in saved_state:
                # previous time steps are cached - no need to recompute
                # key and value if they are static
                if static_kv:
                    assert kv_same and not qkv_same
                    key = value = None
        else:
            saved_state = None

        if qkv_same:
            # self-attention
            q, k, v = self.in_proj_qkv(query)
        elif kv_same:
            # encoder-decoder attention
            q = self.in_proj_q(query)
            if key is None:
                assert value is None
                k = v = None
            else:
                k, v = self.in_proj_kv(key)
        else:
            q = self.in_proj_q(query)
            k = self.in_proj_k(key)
            v = self.in_proj_v(value)
        q *= self.scaling

        if self.bias_k is not None:
            assert self.bias_v is not None
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [key_padding_mask, key_padding_mask.new_zeros(key_padding_mask.size(0), 1)], dim=1)

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        if saved_state is not None:
            # saved states are stored with shape (bsz, num_heads, seq_len, head_dim)
            if 'prev_key' in saved_state:
                prev_key = saved_state['prev_key'].view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    k = prev_key
                else:
                    k = torch.cat((prev_key, k), dim=1)
            if 'prev_value' in saved_state:
                prev_value = saved_state['prev_value'].view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    v = prev_value
                else:
                    v = torch.cat((prev_value, v), dim=1)
            saved_state['prev_key'] = k.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state['prev_value'] = v.view(bsz, self.num_heads, -1, self.head_dim)

            self._set_input_buffer(incremental_state, saved_state)

        src_len = k.size(1)

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        if self.add_zero_attn:
            src_len += 1
            k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [key_padding_mask, torch.zeros(key_padding_mask.size(0), 1).type_as(key_padding_mask)], dim=1)

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            attn_weights += attn_mask

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.float().masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf'),
            ).type_as(attn_weights)  # FP16 support: cast to float and back
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = F.softmax(attn_weights.float(), dim=-1).type_as(attn_weights)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        attn = torch.bmm(attn_weights, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)

        if need_weights:
            # average attention weights over heads
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.sum(dim=1) / self.num_heads
        else:
            attn_weights = None

        return attn, attn_weights

    def in_proj_qkv(self, query):
        return self._in_proj(query).chunk(3, dim=-1)

    def in_proj_kv(self, key):
        return self._in_proj(key, start=self.in_features).chunk(2, dim=-1)

    def in_proj_q(self, query):
        return self._in_proj(query, end=self.in_features)

    def in_proj_k(self, key):
        return self._in_proj(key, start=self.in_features, end=2 * self.in_features)

    def in_proj_v(self, value):
        return self._in_proj(value, start=2 * self.in_features)

    def _in_proj(self, input, start=0, end=None):
        weight = self.in_proj_weight
        bias = self.in_proj_bias
        weight = weight[start:end, :]
        if bias is not None:
            bias = bias[start:end]
        return F.linear(input, weight, bias)

class MultiheadAttentionWithDirectionalMask(MultiheadAttention):

    def __init__(
            self,
            in_features,
            num_heads,
            dropout=0.,
            bias=True,
            add_bias_kv=False,
            add_zero_attn=False,
            mask_backward=False,
            mask_present=False,
            **kwargs):
        super().__init__(
            in_features,
            num_heads,
            dropout,
            bias,
            add_bias_kv,
            add_zero_attn,
            mask_backward=mask_backward,
            mask_present=mask_present,
            **kwargs)
        if mask_present or mask_backward:
            raise NotImplementedError("Not working ATM")
            #TODO: fix
        self.mask_backward = mask_backward
        self.mask_present = mask_present

    def _make_directional_mask(self, q, k):
        mask = self.in_proj_weight.new_ones(q.size(0), k.size(0))
        if self.mask_backward:
            mask = mask.tril(-1)
        else:
            mask = mask.triu(1)
        mask *= -1e9
        return mask

    def forward(self, query, key, value, key_padding_mask=None, incremental_state=None, need_weights=True,
                 static_kv=False, attn_mask=None):
        attn_mask_directional = self._make_directional_mask(query, key)
        if attn_mask is not None:
            attn_mask = attn_mask * attn_mask_directional
        else:
            attn_mask = attn_mask_directional
        attn, attn_weights = super().forward(query, key, value, key_padding_mask, incremental_state, need_weights, static_kv,
                                              attn_mask)
        return attn, attn_weights

if __name__ == "__main__":

    from qbert.nn.adaptive import AdaptiveInput

    embed_dim = 1000
    num_heads = 1
    embedding = AdaptiveInput(1000, embed_dim, cutoffs=[50, 100, 250])
    attnl = MultiheadAttention(embed_dim, num_heads)
    attnl_fw1 = MultiheadAttentionWithDirectionalMask(embed_dim, num_heads)
    #attnl_bw1 = MultiheadAttentionWithDirectionalMask(embed_dim, num_heads, mask_backward=True)
    #attnl_fw2 = MultiheadAttentionWithDirectionalMask(embed_dim, num_heads, mask_present=True)
    #attnl_bw2 = MultiheadAttentionWithDirectionalMask(embed_dim, num_heads, mask_backward=True, mask_present=True)

    x = torch.Tensor([
        [12, 15,  1, 24, 36, 7],
        [ 1, 56, 28, 67,  0, 0],
    ]).long().transpose(0, 1)
    e = embedding(x)

    attn, attn_weights = attnl(e, e, e)
    attn_fw1, attn_weights_fw1 = attnl_fw1(e, e, e)
    print(attn_weights)
    print(attn_weights_fw1)
    #attn_bw1, attn_weights_bw1 = attnl_bw1(e, e, e)
    #attn_fw2, attn_weights_fw2 = attnl_fw2(e, e, e)
    #attn_bw2, attn_weights_bw2 = attnl_bw2(e, e, e)