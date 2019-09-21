from collections import OrderedDict

import torch
from torch.nn import functional as F

from fairseq import options
from fairseq.models import register_model, register_model_architecture
from fairseq.models import BaseFairseqModel, FairseqDecoder
from fairseq.models.transformer import Embedding
from fairseq.modules import CharacterTokenEmbedder, AdaptiveInput

from qbert.fairseq_ext.modules.contextual_embeddings import QBERTEmbedder, BERTEmbedder, ELMOEmbedder, \
    BaseContextualEmbedder, FakeInput, FlairEmbedder


class FairseqTaggerDecoder(FairseqDecoder):
    """Base class for decoders."""

    def __init__(self, dictionary):
        super().__init__(dictionary)
        self.dictionary = dictionary

    def forward(self, tokens, precomputed_embedded=None):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for input feeding/teacher forcing
            encoder_out (Tensor, optional): output from the encoder, used for
                encoder-side attention

        Returns:
            tuple:
                - the last decoder layer's output of shape
                  `(batch, tgt_len, vocab)`
                - the last decoder layer's attention weights of shape
                  `(batch, tgt_len, src_len)`
        """
        raise NotImplementedError

    def get_normalized_probs(self, net_output, log_probs, sample):
        """Get normalized probabilities (or log probs) from a net's output."""

        if hasattr(self, 'adaptive_softmax') and self.adaptive_softmax is not None:
            assert sample is not None and 'target' in sample
            out = self.adaptive_softmax.get_log_prob(net_output[0], sample['target'])
            return out.exp_() if not log_probs else out

        logits = net_output[0].float()
        if log_probs:
            return F.log_softmax(logits, dim=-1)
        else:
            return F.softmax(logits, dim=-1)

    def max_positions(self):
        """Maximum input length supported by the decoder."""
        return 1e6  # an arbitrary large number

    def upgrade_state_dict(self, state_dict):
        """Upgrade a (possibly old) state dict for new versions of fairseq_ext."""
        return state_dict

class TaggerModel(BaseFairseqModel):
    """Base class for sequence labeling models.

    Args:
        decoder (FairseqDecoder): the decoder
    """

    def __init__(self, embedder, decoder, kind, use_all_hidden=False, use_embeddings=True):
        super().__init__()
        self.embedder = embedder
        self.decoder = decoder
        self.kind = kind

        self.use_all_hidden = use_all_hidden
        self.use_embeddings = use_embeddings

        if embedder is not None:
            self.n_hidden_states = (embedder.n_hidden_states if use_all_hidden else 1) + int(use_embeddings)
            if self.n_hidden_states > 1:
                self.layer_weights = torch.nn.Parameter(torch.zeros(self.n_hidden_states))
                self.layer_weights.requires_grad = True
            else:
                self.layer_weights = None
        else:
            self.n_hidden_states = None
            self.layer_weights = None
        assert isinstance(self.decoder, FairseqDecoder)

    @staticmethod
    def add_args(parser):
        parser.add_argument("--context-embeddings", action="store_true",
                            help="The hidden states of a pretrained model as contextual embeddings")
        parser.add_argument("--context-embeddings-type", type=str, choices=['qbert', 'elmo', 'bert', 'flair'], default='qbert')
        parser.add_argument('--context-embeddings-bert-model', type=str, default=BERTEmbedder.DEFAULT_MODEL)
        parser.add_argument('--context-embeddings-elmo-options', type=str, default=ELMOEmbedder.DEFAULT_OPTIONS)
        parser.add_argument('--context-embeddings-elmo-weights', type=str, default=ELMOEmbedder.DEFAULT_WEIGHTS)
        parser.add_argument("--context-embeddings-qbert-checkpoint", type=str, default="", metavar="P",
                            help="The path of the pretrained model checkpoint to use as contextual embeddings generator")
        parser.add_argument('--context-embeddings-flair-forward', type=str, default=FlairEmbedder.DEFAULT_MODEL_FW)
        parser.add_argument('--context-embeddings-flair-backward', type=str, default=FlairEmbedder.DEFAULT_MODEL_BW)
        parser.add_argument('--context-embeddings-flair-embeddings', type=str, default=FlairEmbedder.DEFAULT_MODEL_EMB)
        parser.add_argument("--context-embeddings-use-all-hidden", action="store_true",
                            help="Use a weighted mean of all the hidden states")
        parser.add_argument('--context-embeddings-use-embeddings', action='store_true',
                            help='use word embeddings')
        parser.add_argument('--context-embeddings-normalize-embeddings', action='store_true',
                            help='normalize word embeddings')
        parser.add_argument("--context-embeddings-trainable", action="store_true",
                            help='Train contextual embeddings')
        parser.add_argument('--decoder-input-dim', type=int, metavar='N',
                            help='decoder input dimension')
        parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
                            help='decoder input dimension')
        parser.add_argument('--decoder-embed-pretrained', type=str, metavar='P', default='')
        parser.add_argument('--decoder-embed-pretrained-freeze', action='store_true')
        parser.add_argument('--character-embeddings', default=False, action='store_true',
                            help='if set, uses character embedding convolutions to produce token embeddings')
        parser.add_argument('--character-filters', type=str, metavar='LIST',
                            default='[(1, 64), (2, 128), (3, 192), (4, 256), (5, 256), (6, 256), (7, 256)]',
                            help='size of character embeddings')
        parser.add_argument('--character-embedding-dim', type=int, metavar='N', default=4,
                            help='size of character embeddings')
        parser.add_argument('--char-embedder-highway-layers', type=int, metavar='N', default=2,
                            help='number of highway layers for character token embeddder')
        parser.add_argument('--adaptive-input', default=False, action='store_true',
                            help='if set, uses adaptive input')
        parser.add_argument('--adaptive-input-factor', type=float, metavar='N',
                            help='adaptive input factor')
        parser.add_argument('--adaptive-input-cutoff', metavar='EXPR',
                            help='comma separated list of adaptive input cutoff points.')

    @classmethod
    def build_model_input(cls, args, dictionary):
        # make sure all arguments are present in older fairseq_ext

        args.context_embeddings = getattr(args, 'context_embeddings', False)

        args.max_source_positions = args.tokens_per_sample
        args.max_target_positions = args.tokens_per_sample

        if args.context_embeddings:
            if args.context_embeddings_type == 'qbert':
                embed_tokens = QBERTEmbedder.from_args(args, {"dictionary": dictionary})
            elif args.context_embeddings_type == 'bert':
                assert not args.context_embeddings_use_embeddings
                embed_tokens = BERTEmbedder(args.context_embeddings_bert_model, False)
            elif args.context_embeddings_type == 'elmo':
                embed_tokens = ELMOEmbedder(args.context_embeddings_elmo_options, args.context_embeddings_elmo_weights, False)
            elif args.context_embeddings_type == 'flair':
                embed_tokens = FlairEmbedder(
                    args.context_embeddings_flair_forward,
                    args.context_embeddings_flair_backward,
                    args.context_embeddings_flair_embeddings,
                    False
                )
            else:
                raise NotImplementedError

        elif args.character_embeddings:
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

            def load_pretrained_embedding_from_file(embed_path, dictionary, embed_dim):
                from fairseq import utils

                num_embeddings = len(dictionary)
                padding_idx = dictionary.pad()
                embed_tokens = Embedding(num_embeddings, embed_dim, padding_idx)
                embed_dict = utils.parse_embedding(embed_path)
                utils.print_embed_overlap(embed_dict, dictionary)
                return utils.load_embedding(embed_dict, dictionary, embed_tokens)

            if args.decoder_embed_pretrained:
                embed_tokens = load_pretrained_embedding_from_file(args.decoder_embed_pretrained, dictionary, args.decoder_input_dim)
                if getattr(args, 'decoder_embed_pretrained', False):
                    for par in embed_tokens.parameters():
                        par.requires_grad = False
            else:
                embed_tokens = Embedding(len(dictionary), args.decoder_input_dim, dictionary.pad())

        return embed_tokens

    @classmethod
    def build_model_decoder(cls, args, dictionary, output_dictionary, embed_tokens):
        raise NotImplementedError

    def get_targets(self, sample, net_output):
        """Get targets from either the sample or the net's output."""
        return sample[self.kind]["target"]

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        embedder = cls.build_model_input(args, task.dictionary)
        if isinstance(embedder, BaseContextualEmbedder):
            decoder = cls.build_model_decoder(args, task.dictionary, task.output_dictionary, FakeInput(embedder.embedding_dim))
            inst = cls(embedder, decoder, task.kind, use_embeddings=args.context_embeddings_use_embeddings, use_all_hidden=args.context_embeddings_use_all_hidden)
        else:
            decoder = cls.build_model_decoder(args, task.dictionary, task.output_dictionary, embedder)
            inst = cls(None, decoder, task.kind, use_embeddings=args.context_embeddings_use_embeddings, use_all_hidden=args.context_embeddings_use_all_hidden)
        return inst

    def forward(self, src_tokens=None, src_lengths=None, src_tokens_str=None):
        """
        Run the forward pass for a decoder-only model.

        Feeds a batch of tokens through the decoder to predict the next tokens.

        Args:
            src_tokens (LongTensor): tokens on which to condition the decoder,
                of shape `(batch, tgt_len)`
            src_lengths (LongTensor): source sentence lengths of shape `(batch)`

        Returns:
            the decoder's output, typically of shape `(batch, seq_len, vocab)`
        """
        if self.embedder is None:
            return self.decoder(src_tokens=src_tokens, src_lengths=src_lengths, src_tokens_str=src_tokens_str)
        else:
            out = self.embedder(src_tokens=src_tokens, src_lengths=src_lengths, src_tokens_str=src_tokens_str)
            if self.use_all_hidden:
                states = out["inner_states"]
            else:
                states = out["inner_states"][-1:]
            if self.use_embeddings:
                embedded = out['embedded']
                states = [embedded] + states
            if len(states) > 1:
                stacked = torch.stack(states, dim=0)
                weights = F.softmax(self.layer_weights, 0).view(-1, 1, 1, 1)
                precomputed_embedded = (weights * stacked).sum(0)
            else:
                precomputed_embedded = states[0]
            return self.decoder(src_tokens, precomputed_embedded=precomputed_embedded)

    def max_positions(self):
        """Maximum length supported by the model."""
        return self.decoder.max_positions()

    def remove_head(self):
        """Removes the head of the model (e.g. the softmax layer) to conserve space when it is not needed"""
        raise NotImplementedError()

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        r"""Returns a dictionary containing a whole state of the module.

        Both parameters and persistent buffers (e.g. running averages) are
        included. Keys are corresponding parameter and buffer names.

        Returns:
            dict:
                a dictionary containing a whole state of the module

        Example::

            >>> module.state_dict().keys()
            ['bias', 'weight']

        """
        destination = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        if (self.embedder is not None) and (isinstance(self.embedder, BaseContextualEmbedder)):
            for name in list(destination.keys()):
                if name.startswith('embedder'):
                    del destination[name]

        return destination

    def upgrade_state_dict(self, state, prefix=''):
        if (self.embedder is not None) and (isinstance(self.embedder, BaseContextualEmbedder)):
            self.embedder.state_dict(state, prefix + 'embedder.')
        

def Linear(in_features, out_features, bias=True):
    m = torch.nn.Linear(in_features, out_features, bias)
    torch.nn.init.xavier_uniform_(m.weight)
    if bias:
        torch.nn.init.constant_(m.bias, 0.)
    return m

class LinearDecoder(FairseqTaggerDecoder):

    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(dictionary)
        if hasattr(embed_tokens, "embed_dim"):
            self.input_dim = embed_tokens.embedding_dim
        else:
            self.input_dim = args.decoder_embed_dim

        self.embed_tokens = embed_tokens

        assert args.decoder_layers >= 1
        self.linears = torch.nn.ModuleList()
        self.dropout = torch.nn.Dropout(args.dropout)
        for i in range(args.decoder_layers - 1):
            in_features = self.input_dim if i == 0 else args.decoder_embed_dim
            out_features = args.decoder_embed_dim
            self.linears.append(Linear(in_features, out_features))
        self.logits = torch.nn.Linear(
            self.input_dim if args.decoder_layers == 1 else args.decoder_embed_dim,
            len(dictionary)
        )

    def embedding_forward(self, tokens):
        out = self.embed_tokens(tokens)
        return out

    def head_forward(self, embedded):
        out = embedded
        for lin in self.linears:
            out = lin(out)
            out = self.dropout(out)
            out = F.relu(out)
        return self.logits(out)

    def forward(self, tokens, precomputed_embedded=None):
        if precomputed_embedded is None:
            out = self.embedding_forward(tokens)
        else:
            out = precomputed_embedded
        out = self.head_forward(out)
        return out, {}

@register_model('linear_seq')
class LinearTaggerModel(TaggerModel):

    @classmethod
    def build_model_decoder(cls, args, dictionary, output_dictionary, embed_tokens):
        return LinearDecoder(args, dictionary=output_dictionary, embed_tokens=embed_tokens)

    @staticmethod
    def add_args(parser):
        TaggerModel.add_args(parser)
        parser.add_argument('--dropout', default=0.1, type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--decoder-layers', type=int, metavar='N',
                            help='num decoder layers')

@register_model_architecture('linear_seq', 'linear_seq')
def linear_seq(args):

    args.context_embeddings_use_all_hidden = getattr(args, "context_embeddings_use_all_hidden", False)
    args.context_embeddings_use_embeddings = getattr(args, "context_embeddings_use_embeddings", False)
    args.context_embeddings_normalize_embeddings = getattr(args, "context_embeddings_normalize_embeddings", True)
    args.context_embeddings_trainable = getattr(args, "context_embeddings_trainable", False)

    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 512)
    args.decoder_layers = getattr(args, 'decoder_layers', 3)
    args.criterion = getattr(args, 'criterion', "weighted_cross_entropy")






