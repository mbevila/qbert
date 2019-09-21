from torch.nn import functional as F

from fairseq.models import register_model, register_model_architecture
from fairseq.models.transformer import TransformerDecoder

from qbert.fairseq_ext.models.sequence_tagging import TaggerModel

class TransformerDecoderNoMask(TransformerDecoder):
    """
    Modified Transformer Decoder for the use with Embedder classes
    """


    def forward(self, src_tokens, src_lengths=None, src_tokens_str=None, precomputed_embedded=None, encoder_out=None, incremental_state=None):
        """
        Args:
            src_tokens (LongTensor): previous decoder outputs of shape
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

        self_attn_padding_mask = (src_tokens == self.embed_tokens.padding_idx).byte()

        # embed positions
        positions = self.embed_positions(
            src_tokens,
            incremental_state=incremental_state,
        ) if self.embed_positions is not None else None

        if incremental_state is not None:
            src_tokens = src_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        x = self.embed_tokens(src_tokens) if precomputed_embedded is None else precomputed_embedded
        # embed tokens and positions
        x = self.embed_scale * x

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        attn = None

        inner_states = [x]

        # decoder layers
        for layer in self.layers:
            x, attn = layer(
                x,
                encoder_out['encoder_out'] if encoder_out is not None else None,
                encoder_out['encoder_padding_mask'] if encoder_out is not None else None,
                incremental_state,
                self_attn_mask=None,
                self_attn_padding_mask=self_attn_padding_mask
            )
            inner_states.append(x)

        if self.normalize:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        if self.adaptive_softmax is None:
            # project back to size of vocabulary
            if self.share_input_output_embed:
                x = F.linear(x, self.embed_tokens.weight)
            else:
                x = F.linear(x, self.embed_out)

        return x, {'attn': attn, 'inner_states': inner_states}


@register_model('transformer_seq')
class TransformerTagger(TaggerModel):

    @staticmethod
    def add_args(parser):
        TaggerModel.add_args(parser)
        parser.add_argument('--dropout', default=0.1, type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--attention-dropout', default=0., type=float, metavar='D',
                            help='dropout probability for attention weights')
        parser.add_argument('--relu-dropout', default=0., type=float, metavar='D',
                            help='dropout probability after ReLU in FFN')
        parser.add_argument('--decoder-output-dim', type=int, metavar='N',
                            help='decoder output dimension')
        parser.add_argument('--decoder-ffn-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension for FFN')
        parser.add_argument('--decoder-layers', type=int, metavar='N',
                            help='num decoder layers')
        parser.add_argument('--decoder-attention-heads', type=int, metavar='N',
                            help='num decoder attention heads')
        parser.add_argument('--decoder-normalize-before', default=False, action='store_true',
                            help='apply layernorm before each decoder block')
        parser.add_argument('--no-token-positional-embeddings', default=False, action='store_true',
                            help='if set, disables positional embeddings (outside self attention)')
        parser.add_argument('--decoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the decoder')

    @classmethod
    def build_model_decoder(cls, args, dictionary, output_dictionary, embed_tokens):
        decoder = TransformerDecoderNoMask(args, output_dictionary, embed_tokens, no_encoder_attn=True,
                                           final_norm=False)
        return decoder

@register_model_architecture('transformer_seq', 'transformer_seq')
def transformer_seq(args):

    args.share_decoder_input_output_embed = getattr(args, 'share_decoder_input_output_embed', False)

    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 512)
    args.decoder_output_dim = getattr(args, 'decoder_output_dim', args.decoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 2048)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 8)
    args.no_token_positional_embeddings = getattr(args, 'no_token_positional_embeddings', True)
    args.decoder_learned_pos = getattr(args, 'decoder_learned_pos', False)

    args.adaptive_softmax_cutoff = getattr(args, 'adaptive_softmax_cutoff', None)

    args.context_embeddings_use_all_hidden = getattr(args, "context_embeddings_use_all_hidden", False)
    args.context_embeddings_use_embeddings = getattr(args, "context_embeddings_use_embeddings", False)
    args.context_embeddings_normalize_embeddings = getattr(args, "context_embeddings_normalize_embeddings", True)
    args.context_embeddings_trainable = getattr(args, "context_embeddings_trainable", False)

    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 512)
    args.decoder_layers = getattr(args, 'decoder_layers', 3)
    args.criterion = getattr(args, 'criterion', "weighted_cross_entropy")