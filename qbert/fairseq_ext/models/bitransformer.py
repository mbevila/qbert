import torch
from torch import nn
from torch.nn import functional as F

from qbert.nn.attentive.submodules import TransformerBlock

from fairseq import options

from fairseq.models import register_model, register_model_architecture, FairseqLanguageModel
from fairseq.models.transformer import TransformerLanguageModel, TransformerDecoder, TransformerDecoderLayer,\
    base_lm_architecture, Embedding, LayerNorm, Linear
from fairseq.modules import CharacterTokenEmbedder, AdaptiveInput, MultiheadAttention

#TODO make a cleaner class to replace TransformerDecoderLayer
class BiTransformerDecoderLayer(nn.Module):
    """Decoder layer block.

    In the original paper each operation (multi-head attention, encoder
    attention or FFN) is postprocessed with: `dropout -> add residual ->
    layernorm`. In the tensor2tensor code they suggest that learning is more
    robust when preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.decoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        no_encoder_attn (bool, optional): whether to attend to encoder outputs.
            Default: ``False``
    """

    def __init__(self, args):
        super().__init__()
        self.embedding_dim = args.decoder_embed_dim
        self.self_attn1 = MultiheadAttention(
            self.embedding_dim, args.decoder_attention_heads,
            dropout=args.attention_dropout,
        )
        self.self_attn2 = MultiheadAttention(
            self.embedding_dim, args.decoder_attention_heads,
            dropout=args.attention_dropout,
        )
        self.dropout = args.dropout
        self.relu_dropout = args.relu_dropout
        self.normalize_before = args.decoder_normalize_before

        self.self_attn_layer_norm_1 = LayerNorm(self.embedding_dim)
        self.self_attn_layer_norm_2 = LayerNorm(self.embedding_dim)
        self.self_attn_layer_norm = LayerNorm(self.embedding_dim * 2)

        self.fc1 = Linear(self.embedding_dim, args.decoder_ffn_embed_dim)
        self.fc2 = Linear(args.decoder_ffn_embed_dim, self.embedding_dim)

        self.final_layer_norm = LayerNorm(self.embedding_dim)

        self.need_attn = True

        self.onnx_trace = False

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def forward(self, x1, x2,
                self_attn_mask=None, self_attn_mask_2=None,
                self_attn_padding_mask=None, self_attn_padding_mask_2=None):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.

        Returns:
            encoded output of shape `(batch, src_len, embed_dim)`
        """

        self_attn_mask_2 = self_attn_mask if self_attn_mask_2 is None else self_attn_mask_2

        if self_attn_padding_mask is not None:
            self_attn_padding_mask_for_emb = self_attn_padding_mask.transpose(0, 1).unsqueeze(-1)

        if self_attn_padding_mask_2 is not None:
            self_attn_padding_mask_2_for_emb = self_attn_padding_mask_2.transpose(0, 1).unsqueeze(-1)

        residual1 = x1
        residual2 = x2
        x1_orig = self.maybe_layer_norm(self.self_attn_layer_norm_1, x1, before=True)
        x2_orig = self.maybe_layer_norm(self.self_attn_layer_norm_2, x2, before=True)

        x1, _ = self.self_attn1(
            query=x1_orig,
            key=x2_orig,
            value=x2_orig,
            key_padding_mask=self_attn_padding_mask,
            need_weights=False,
            attn_mask=self_attn_mask,
        )

        if self_attn_padding_mask is not None:
            x1[self_attn_padding_mask_for_emb.expand_as(x1)] = 0.

        x2, _ = self.self_attn1(
            query=x2_orig,
            key=x1_orig,
            value=x1_orig,
            key_padding_mask=self_attn_padding_mask_2,
            need_weights=False,
            attn_mask=self_attn_mask_2,
        )

        if self_attn_padding_mask_2 is not None:
            x2[self_attn_padding_mask_2_for_emb.expand_as(x1)] = 0.

        x1 = F.dropout(x1, p=self.dropout, training=self.training) + residual1
        x2 = F.dropout(x2, p=self.dropout, training=self.training) + residual2
        x = x1 + x2
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, after=True)

        attn = None

        residual = x
        x = self.maybe_layer_norm(self.final_layer_norm, x, before=True)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.relu_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(self.final_layer_norm, x, after=True)
        if self.onnx_trace:
            saved_state = self.self_attn._get_input_buffer(incremental_state)
            self_attn_state = saved_state["prev_key"], saved_state["prev_value"]
            return x, attn, self_attn_state
        return x, attn

    def maybe_layer_norm(self, layer_norm, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return layer_norm(x)
        else:
            return x

    def make_generation_fast_(self, need_attn=False, **kwargs):
        self.need_attn = need_attn

@register_model('transformer_bilm')
class BiTransformerLanguageModel(FairseqLanguageModel):

    def __init__(self, decoder):
        super().__init__(decoder)

    @staticmethod
    def add_args(parser):
        TransformerLanguageModel.add_args(parser)
        parser.add_argument("--decoder-use-biattention", action="store_true", help=
            "Allow as a last step the left to right to attend over the right to left transformer layer, and vice versa")
        parser.add_argument("--decoder-share-directions", action="store_true")

    @property
    def supported_targets(self):
        return {"self"}


    @classmethod
    def build_model_decoder(cls, args, dictionary, output_dictionary=None):

        if output_dictionary is None:
            output_dictionary = dictionary

        if args.character_embeddings:
            embed_tokens = CharacterTokenEmbedder(dictionary, eval(args.character_filters),
                                                  args.character_embedding_dim,
                                                  args.decoder_embed_dim,
                                                  args.char_embedder_highway_layers,
                                                  )
        elif args.adaptive_input:
            embed_tokens = AdaptiveInput(len(dictionary), dictionary.pad(), args.decoder_input_dim,
                                         args.adaptive_input_factor, args.decoder_embed_dim,
                                         options.eval_str_list(args.adaptive_input_cutoff, type=int))
        else:
            embed_tokens = Embedding(len(dictionary), args.decoder_input_dim, dictionary.pad())

        return BiTransformerDecoder(args, output_dictionary, embed_tokens, no_encoder_attn=True, final_norm=False)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older fairseq_ext
        base_lm_architecture(args)

        if hasattr(args, 'no_tie_adaptive_proj') and args.no_tie_adaptive_proj == False:
            # backward compatibility
            args.tie_adaptive_proj = True

        if not hasattr(args, 'max_source_positions'):
            args.max_source_positions = args.tokens_per_sample
        if not hasattr(args, 'max_target_positions'):
            args.max_target_positions = args.tokens_per_sample

        if args.tie_adaptive_weights:
            assert args.adaptive_input
            assert args.adaptive_input_factor == args.adaptive_softmax_factor
            assert args.adaptive_softmax_cutoff == args.adaptive_input_cutoff, '{} != {}'.format(
                args.adaptive_softmax_cutoff, args.adaptive_input_cutoff)
            assert args.decoder_input_dim == args.decoder_output_dim

        decoder = cls.build_model_decoder(args, task.dictionary, task.output_dictionary)

        return cls(decoder)

    @property
    def n_hidden_states(self):
        return self.decoder.n_hidden_states


class BiTransformerDecoder(TransformerDecoder):

    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False, left_pad=False, final_norm=True):
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn, left_pad, final_norm)

        self.share_directions = False if not hasattr(args, 'decoder_share_directions') else args.decoder_share_directions
        if self.share_directions:
            self.layers_bw = None
        else:
            self.layers_bw = nn.ModuleList([])
            self.layers_bw.extend([
                TransformerDecoderLayer(args, no_encoder_attn)
                for _ in range(args.decoder_layers)
            ])

        self.padding_idx = dictionary.pad()

        self.embedding_dim = self.embed_dim = args.decoder_embed_dim
        self.n_hidden_states = args.decoder_layers
        if args.decoder_use_biattention:
            self.n_hidden_states += 1

        self.use_biattention = args.decoder_use_biattention
        if self.use_biattention:
            self.biblock = BiTransformerDecoderLayer(args)
        else:
            self.biblock = None

        self.pad_fw = nn.Parameter(torch.randn(1, 1, args.decoder_embed_dim), requires_grad=True)
        self.pad_bw = nn.Parameter(torch.randn(1, 1, args.decoder_embed_dim), requires_grad=True)

    def buffered_past_mask(self, tensor):
        return self.buffered_future_mask(tensor).t_()

    def compute_hidden_states(self, tokens, batch_major=False):
        encoder_out = None
        incremental_state = None

        if self.training:
            padding_mask_bm = None
        else:
            padding_mask_bm = (tokens == self.padding_idx).byte()
            if padding_mask_bm is not None:
                padding_mask_for_emb = padding_mask_bm.transpose(0,1).unsqueeze(-1)

        # embed positions
        positions = self.embed_positions(
            tokens,
            incremental_state=incremental_state,
        ) if self.embed_positions is not None else None

        if incremental_state is not None:
            tokens = tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # embed tokens and positions
        embedded = self.embed_tokens(tokens)
        x = self.embed_scale * embedded

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        attn = None

        inner_states = []

        # decoder layers
        x_fw = x
        for layer in self.layers:
            x_fw, attn = layer(
                x_fw,
                encoder_out['encoder_out'] if encoder_out is not None else None,
                encoder_out['encoder_padding_mask'] if encoder_out is not None else None,
                incremental_state,
                self_attn_mask=self.buffered_future_mask(x) if incremental_state is None else None,
                self_attn_padding_mask=padding_mask_bm,
            )
            inner_states.append(x_fw)
        x_bw = x
        layers_bw = self.layers if self.share_directions else self.layers_bw
        for layer in layers_bw:
            x_bw, attn = layer(
                x_bw,
                encoder_out['encoder_out'] if encoder_out is not None else None,
                encoder_out['encoder_padding_mask'] if encoder_out is not None else None,
                incremental_state,
                self_attn_mask=self.buffered_past_mask(x) if incremental_state is None else None,
                self_attn_padding_mask=padding_mask_bm,
            )
            if padding_mask_bm is not None:
                x_bw[padding_mask_for_emb.expand_as(x_bw)] = 0.
            inner_states[-1] = inner_states[-1] + x_bw

        pad_fw = self.pad_fw.expand(1, x.size(1), self.pad_fw.size(2))
        pad_bw = self.pad_bw.expand(1, x.size(1), self.pad_bw.size(2))
        x_fw = torch.cat([pad_fw, x_fw[:-1]], dim=0)
        x_bw = torch.cat([x_bw[1:], pad_bw], dim=0)

        if padding_mask_bm is not None:
            self_attn_padding_mask = torch.cat([ # relative to k, so x_bw in this case
                padding_mask_bm[:, 1:], padding_mask_bm.new_zeros(padding_mask_bm.size(0), 1)
            ], dim=1)
            self_attn_padding_mask_2 = torch.cat([ # relative to k, so x_bw in this case
                padding_mask_bm.new_zeros(padding_mask_bm.size(0), 1), padding_mask_bm[:, :-1]
            ], dim=1)
        else:
            self_attn_padding_mask = self_attn_padding_mask_2 = None

        if self.use_biattention:
            x, attn = self.biblock(
                x_fw, x_bw,
                self_attn_mask=self.buffered_past_mask(x_bw),
                self_attn_mask_2=self.buffered_future_mask(x_fw),
                self_attn_padding_mask=self_attn_padding_mask,
                self_attn_padding_mask_2=self_attn_padding_mask_2
            )
        else:
            x = x_fw + x_bw

        if self.normalize:
            x = self.layer_norm(x)

        inner_states.append(x)

        if batch_major:
            inner_states = [x.transpose(0, 1) for x in inner_states]

        return {'attn': attn, 'inner_states': inner_states, 'embedded': embedded}

    def forward(self, prev_output_tokens, encoder_out=None, incremental_state=None):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for input feeding/teacher forcing
            encoder_out (Tensor, optional): output from the encoder, used for
                encoder-side attention
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`

        Returns:
            tuple:
                - the last decoder layer's output of shape `(batch, tgt_len,
                  vocab)`
                - the last decoder layer's attention weights of shape `(batch,
                  tgt_len, src_len)`
        """

        data = self.compute_hidden_states(prev_output_tokens, batch_major=False)

        # T x B x C -> B x T x C
        x = data['inner_states'][-1].transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        if self.adaptive_softmax is None:
            # project back to size of vocabulary
            if self.share_input_output_embed:
                x = F.linear(x, self.embed_tokens.weight)
            else:
                x = F.linear(x, self.embed_out)

        return x, data


@register_model_architecture('transformer_bilm', 'transformer_bilm')
def transformer_bilm(args):

    args.self_target = getattr(args, 'self_target', True)
    args.past_target = getattr(args, 'past_target', False)
    args.past_target = getattr(args, 'future_target', False)

    args.character_embeddings = getattr(args, 'character_embeddings', False)

    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 512)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 2048)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 8)
    args.decoder_learned_pos = getattr(args, 'decoder_learned_pos', False)

    args.decoder_output_dim = getattr(args, 'decoder_output_dim', args.decoder_embed_dim)
    args.decoder_input_dim = getattr(args, 'decoder_input_dim', args.decoder_embed_dim)
    args.decoder_use_biattention = getattr(args, 'decoder_use_biattention', False)
    args.decoder_share_directions = getattr(args, 'decoder_share_directions', False)

    args.criterion = getattr(args, 'criterion', "adaptive_loss")
    args.adaptive_input = getattr(args, 'adaptive_input', False)
    args.adaptive_input_factor = getattr(args, 'adaptive_input_factor', 4)
    args.adaptive_input_cutoff = getattr(args, 'adaptive_input_cutoff', None)
    args.adaptive_softmax_factor = getattr(args, 'adaptive_softmax_factor', args.adaptive_input_factor)
    args.adaptive_softmax_cutoff = getattr(args, 'adaptive_softmax_cutoff', args.adaptive_input_cutoff)
    args.adaptive_softmax_dropout = getattr(args, 'adaptive_softmax_dropout', 0.3)
    args.tie_adaptive_weights = getattr(args, 'tie_adaptive_weights', False)
    args.tie_adaptive_proj = getattr(args, 'tie_adaptive_proj', False)

    # The model training is not stable without this
    args.decoder_normalize_before = True


@register_model_architecture('transformer_bilm', 'transformer_bilm_big')
def transformer_bilm_big(args):
    args.adaptive_input_cutoff = getattr(args, 'adaptive_input_cutoff', [10_000, 20_000, 100_000])
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 1024)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 4096)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 16)
    transformer_bilm(args)


@register_model_architecture('transformer_bilm', 'transformer_bilm_wiki103')
def transformer_bilm_wiki103(args):
    args.adaptive_input_cutoff = getattr(args, 'adaptive_input_cutoff', [10_000, 20_000, 100_000])
    args.dropout = getattr(args, 'dropout', 0.3)
    transformer_bilm_big(args)


@register_model_architecture('transformer_bilm', 'transformer_bilm_biattentive')
def transformer_bilm_biattentive(args):
    args.decoder_use_biattention = getattr(args, 'decoder_use_biattention', True)
    transformer_bilm(args)


@register_model_architecture('transformer_bilm', 'transformer_bilm_biattentive_big')
def transformer_bilm_biattentive_big(args):
    args.adaptive_input_cutoff = getattr(args, 'adaptive_input_cutoff', [10_000, 20_000, 100_000])
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 1024)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 4096)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 16)
    transformer_bilm_biattentive(args)


@register_model_architecture('transformer_bilm', 'transformer_bilm_biattentive_wiki103')
def transformer_bilm_biattentive_wiki103(args):
    args.adaptive_input_cutoff = getattr(args, 'adaptive_input_cutoff', [10_000, 20_000, 100_000])
    args.dropout = getattr(args, 'dropout', 0.3)
    transformer_bilm_biattentive_big(args)
