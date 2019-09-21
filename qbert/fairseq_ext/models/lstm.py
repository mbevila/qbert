import torch
from fairseq import utils, options
from fairseq.models import register_model, register_model_architecture
from fairseq.models.lstm import LSTM, Linear
from fairseq.modules import AdaptiveSoftmax
from torch.nn import functional as F

from qbert.fairseq_ext.models.sequence_tagging import FairseqTaggerDecoder, TaggerModel


class LSTMTaggerDecoder(FairseqTaggerDecoder):

    def __init__(
        self, dictionary, embed_tokens, embed_dim=512,
        num_layers=1, dropout_in=0.1, dropout_out=0.1, bidirectional=False,
        left_pad=False, padding_value=0.
    ,   adaptive_softmax=False, adaptive_softmax_cutoff=[],
        adaptive_softmax_dropout=0.1, adaptive_softmax_factor=None
    ):
        super(LSTMTaggerDecoder, self).__init__(dictionary=dictionary)

        if hasattr(embed_tokens, "embedded_dim"):
            self.in_embed_dim = embed_tokens.embedded_dim
        elif hasattr(embed_tokens, "embed_dim"):
            self.in_embed_dim = embed_tokens.embed_dim
        elif hasattr(embed_tokens, "embedding_dim"):
            self.in_embed_dim = embed_tokens.embedding_dim
        else:
            raise Exception
        self.output_units = self.embed_dim = embed_dim
        self.out_embed_dim = len(dictionary)

        self.num_layers = num_layers
        self.dropout_in = dropout_in
        self.dropout_out = dropout_out

        self.bidirectional = bidirectional
        if self.bidirectional:
            #self.output_units *= 2
            pass

        self.padding_idx = dictionary.pad()
        self.padding_value = 0.
        self.left_pad = left_pad

        self.embed_tokens = embed_tokens

        self.fc_in = self.fc_out1 = self.fc_out2 = None
        if self.in_embed_dim != self.embed_dim:
            self.fc_in = Linear(self.in_embed_dim, self.embed_dim)
        if self.output_units != self.embed_dim:
            self.fc_out1 = Linear(self.output_units, self.embed_dim)
        if self.embed_dim != self.out_embed_dim:
            self.fc_out2 = Linear(self.embed_dim, self.out_embed_dim)

        self.lstm = LSTM(
            input_size=embed_dim,
            hidden_size=embed_dim,
            num_layers=num_layers,
            dropout=self.dropout_out if num_layers > 1 else 0.,
            bidirectional=bidirectional,
        )

        self.adaptive_softmax = None

        if adaptive_softmax:
            self.adaptive_softmax = AdaptiveSoftmax(
                len(dictionary),
                self.embed_dim,
                adaptive_softmax_cutoff,
                dropout=adaptive_softmax_dropout,
                adaptive_inputs=None,
                factor=adaptive_softmax_factor,
                tie_proj=False,
            )

    def forward(self, tokens, lengths=None, precomputed_embedded=None, **kwargs):

        bsz, seqlen = tokens.size()

        if self.left_pad:
            # convert left-padding to right-padding
            tokens = utils.convert_padding_direction(
                tokens,
                self.padding_idx,
                left_to_right=True,
            )
        if lengths is None:
            lengths = (tokens != self.padding_idx).sum(1)

        if precomputed_embedded is None:
            x = self.embed_tokens(tokens)
        else:
            x = precomputed_embedded
        x = F.dropout(x, p=self.dropout_in, training=self.training)

        if self.fc_in:
            x = self.fc_in(x)

        # sorting sequences by len otherwise pack_padded_sequence will complain
        lengths_sorted, perm_index = lengths.sort(0, descending=True)
        if (lengths_sorted != lengths).sum():
            needs_perm = True
            x = x[perm_index]
            lengths = lengths_sorted
        else:
            needs_perm = False

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        packed_x = torch.nn.utils.rnn.pack_padded_sequence(x, lengths.data.tolist())
        # apply LSTM
        if self.bidirectional:
            state_size = 2 * self.num_layers, bsz, self.embed_dim
        else:
            state_size = self.num_layers, bsz, self.embed_dim
        h0 = x.data.new(*state_size).zero_()
        c0 = x.data.new(*state_size).zero_()
        packed_outs, (final_hiddens, final_cells) = self.lstm(packed_x, (h0, c0))

        # unpack outputs and apply dropout
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_outs, padding_value=self.padding_value)

        x = F.dropout(x, p=self.dropout_out, training=self.training)
        #assert list(x.size()) == [seqlen, bsz, self.output_units]

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        # restoring original order
        if needs_perm:
            odx = perm_index.view(-1, 1).unsqueeze(1).expand_as(x)
            x = x.gather(0, odx)

        if self.bidirectional:

            def combine_bidir(outs):
                return outs.view(self.num_layers, 2, bsz, -1).transpose(1, 2).contiguous().view(self.num_layers, bsz, -1)

            final_hiddens = combine_bidir(final_hiddens)
            final_cells = combine_bidir(final_cells)

            x = x.view(x.size(0), x.size(1), 2, -1).sum(2)

        if self.fc_out1 is not None:
            x = self.fc_out1(x)

        if self.adaptive_softmax is None and self.fc_out2 is not None :
            x = self.fc_out2(x)

        return x, {'hidden_states': (final_hiddens, final_cells)}

@register_model('lstm_seq')
class LSTMTaggerModel(TaggerModel):

    @classmethod
    def build_model_decoder(cls, args, dictionary, output_dictionary, embed_tokens):
        decoder = LSTMTaggerDecoder(
            dictionary=output_dictionary,
            embed_tokens=embed_tokens,
            embed_dim=args.decoder_embed_dim,
            num_layers=args.decoder_layers,
            dropout_in=args.dropout_in,
            dropout_out=args.dropout,
            bidirectional=True,
            left_pad=False,
            padding_value=0.,
            adaptive_softmax=args.criterion == 'adaptive_loss',
            adaptive_softmax_cutoff=
                options.eval_str_list(args.adaptive_softmax_cutoff, type=int)
                    if args.criterion == 'adaptive_loss'
                    else None,
            adaptive_softmax_dropout=args.adaptive_softmax_dropout,
            adaptive_softmax_factor=args.adaptive_softmax_factor
        )
        return decoder

    @staticmethod
    def add_args(parser):
        TaggerModel.add_args(parser)
        parser.add_argument('--dropout', default=0.1, type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--dropout-in', default=0.1, type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--decoder-layers', type=int, default=3, metavar='N',
                            help='num decoder layers')
        parser.add_argument('--adaptive-softmax', default=False, action='store_true',
                            help='if set, uses adaptive softmax')
        parser.add_argument('--adaptive-softmax-dropout', default=0.1, type=float, metavar='D',
                            help='adaptive softmax dropout probability')
        parser.add_argument('--adaptive-softmax-factor', type=int, default=4, metavar='N',
                            help='adaptive softmax factor')
        parser.add_argument('--adaptive-softmax-cutoff', default="5000,125000", metavar='EXPR',
                            help='comma separated list of adaptive input cutoff points.')

@register_model_architecture('lstm_seq', 'lstm_seq')
def lstm_seq(args):

    args.context_embeddings_use_all_hidden = getattr(args, "context_embeddings_use_all_hidden", False)
    args.context_embeddings_use_embeddings = getattr(args, "context_embeddings_use_embeddings", False)
    args.context_embeddings_normalize_embeddings = getattr(args, "context_embeddings_normalize_embeddings", True)
    args.context_embeddings_trainable = getattr(args, "context_embeddings_trainable", False)

    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 512)
    args.decoder_layers = getattr(args, 'decoder_layers', 3)
    args.criterion = getattr(args, 'criterion', "weighted_cross_entropy")




