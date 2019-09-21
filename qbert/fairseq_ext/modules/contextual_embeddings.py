import abc
import enum
from typing import Union, List, Tuple

import torch
from torch.nn import functional as F
from fairseq.models.transformer import LayerNorm

from qbert.fairseq_ext.models.bitransformer import BiTransformerLanguageModel


class FakeInput(torch.nn.Module):

    def __init__(self, embedding_dim, padding_idx=1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.embed_dim = embedding_dim
        self.padding_idx = padding_idx


class BaseContextualEmbedder(torch.nn.Module, metaclass=abc.ABCMeta):

    embedding_dim: int
    n_hidden_states: int
    retrain_model: bool

    def __init__(self, retrain_model: bool = False):
        super().__init__()
        self.retrain_model = retrain_model

    def forward(
            self,
            src_tokens: Union[None, torch.LongTensor] = None,
            src_tokens_str: Union[None, List[List[str]]] = None,
            batch_major: bool = True,
            **kwargs
    ):
        pass

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def is_cuda(self):
        return next(self.parameters()).is_cuda

    @property
    def embed_dim(self):
        return self.embedding_dim

    @property
    def embedded_dim(self):
        return self.embedding_dim

class ELMOEmbedder(BaseContextualEmbedder):

    DEFAULT_OPTIONS = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
    DEFAULT_WEIGHTS = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

    @staticmethod
    def _do_imports():
        from allennlp.modules import elmo
        from allennlp.nn.util import remove_sentence_boundaries
        return elmo, remove_sentence_boundaries

    def __init__(self, options_file: Union[None, str]=None, weight_file: Union[None, str]=None, retrain_model=False):
        super(ELMOEmbedder, self).__init__(retrain_model=retrain_model)

        assert not retrain_model

        if not options_file:
            options_file = self.DEFAULT_OPTIONS
        if not weight_file:
            weight_file = self.DEFAULT_WEIGHTS

        elmo, self.elmo_remove_sentence_boundaries = self._do_imports()
        self.elmo_bilm = elmo._ElmoBiLm(options_file, weight_file)
        self.elmo_batch_to_ids = elmo.batch_to_ids
        self.embedding_dim = self.elmo_bilm._elmo_lstm.hidden_size

        for par in self.parameters():
            par.requires_grad = False


    def forward(self, src_tokens_str: List[List[str]], src_tokens=None, batch_major=True, padding_str='<pad>',**kwargs):
        """
        Modified version of allennlp's ElmoEmbedder.batch_to_embeddings method

        """
        assert src_tokens_str is not None

        src_tokens_str = [[t for t in seq if t != padding_str] for seq in src_tokens_str]

        character_ids = self.elmo_batch_to_ids(src_tokens_str)
        character_ids = character_ids.to(self.device)

        bilm_output = self.elmo_bilm(character_ids)
        layer_activations = bilm_output['activations']
        mask_with_bos_eos = bilm_output['mask']

        # without_bos_eos is a 3 element list of (activation, mask) tensor pairs,
        # each with size (batch_size, num_timesteps, dim and (batch_size, num_timesteps)
        # respectively.
        without_bos_eos = [self.elmo_remove_sentence_boundaries(layer, mask_with_bos_eos)[0]
                           for layer in layer_activations]
        # Converts a list of pairs (activation, mask) tensors to a single tensor of activations.
        without_bos_eos = [t.view(t.size(0), t.size(1), 2, -1).sum(2) for t in without_bos_eos]

        output = {'embedded': without_bos_eos[0], 'inner_states': without_bos_eos[1:]}
        return output

    @property
    def n_hidden_states(self):
        return self.elmo_bilm._token_embedder.get_output_dim()

class FlairEmbedder(BaseContextualEmbedder):

    DEFAULT_MODEL_FW = 'mix-forward'
    DEFAULT_MODEL_BW = 'mix-backward'
    DEFAULT_MODEL_EMB = 'glove'
    @staticmethod
    def _do_imports():
        from flair.data import Sentence
        from flair.embeddings import FlairEmbeddings, WordEmbeddings, StackedEmbeddings
        return Sentence, FlairEmbeddings, WordEmbeddings, StackedEmbeddings

    def __init__(self, name_fw: Union[str, None] = None, name_bw: Union[str, None] = None, name_emb=None, retrain_model=False):
        assert not retrain_model
        super(FlairEmbedder, self).__init__(retrain_model=False)

        assert not retrain_model

        if not name_fw:
            name_fw = self.DEFAULT_MODEL_FW

        if not name_bw:
            name_bw = self.DEFAULT_MODEL_BW

        if not name_emb:
            name_emb = self.DEFAULT_MODEL_EMB

        self._sentence_cls, _model_cls, _embedding_cls, _stacked_cls = self._do_imports()
        model_fw = _model_cls(name_fw)
        model_bw = _model_cls(name_bw)
        model_emb = _embedding_cls(name_emb)
        self.model = _stacked_cls([model_fw, model_bw, model_emb]).cuda()

        for par in self.parameters():
            par.requires_grad = False

    def forward(self, src_tokens_str: List[List[str]], src_tokens=None, batch_major=True, padding_str='<pad>',
                **kwargs):

        assert src_tokens_str is not None

        sentences = []
        for sent in src_tokens_str:
            sent = [t.replace(' ', '_') for t in sent if t != '<pad>']
            sent_obj = self._sentence_cls(" ".join(sent), use_tokenizer=False)
            if len(sent) != len(sent_obj):
                print(sent)
                print(sent_obj)
                for t1, t2 in zip(sent, sent_obj):
                    print(t1, end="\t")
                    print(t2)
                raise ValueError
            sentences.append(sent_obj)

        maxlen = max([len(x) for x in src_tokens_str])

        self.model.embed(sentences)
        sentence_vectors = []
        for sent in sentences:
            sent_v = torch.stack([t.embedding for t in sent], dim=0)
            if sent_v.shape[0] < maxlen:
                sent_v = torch.cat([sent_v, torch.zeros(maxlen - sent_v.shape[0], sent_v.shape[1])])
            sentence_vectors.append(sent_v)
        sentence_vectors = torch.stack(sentence_vectors, dim=0).to(self.device)

        return {'inner_states': [sentence_vectors]}

    @property
    def embedding_dim(self):
        return self.model.embedding_length

    @property
    def n_hidden_states(self):
        return 1

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def is_cuda(self):
        return next(self.parameters()).is_cuda



class BERTEmbedder(BaseContextualEmbedder):

    DEFAULT_MODEL = 'bert-base-cased'

    @staticmethod
    def _do_imports():
        import pytorch_pretrained_bert as bert
        return bert

    def __init__(self, name: Union[str, None] = None, retrain_model=False):
        assert not retrain_model
        super(BERTEmbedder, self).__init__(retrain_model=False)

        assert not retrain_model

        if not name:
            name = self.DEFAULT_MODEL

        bert = self._do_imports()
        self.bert_tokenizer = bert.BertTokenizer.from_pretrained(name)
        self.bert_model = bert.BertModel.from_pretrained(name)

        for par in self.parameters():
            par.requires_grad = False

    def _subtokenize_sequence(self, tokens):
        split_tokens = []
        merge_to_previous = []
        for token in tokens:
            for i, sub_token in enumerate(self.bert_tokenizer.wordpiece_tokenizer.tokenize(token)):
                split_tokens.append(sub_token)
                merge_to_previous.append(i > 0)

        split_tokens = ['[CLS]'] + split_tokens + ['[SEP]']
        return split_tokens, merge_to_previous

    def _convert_to_indices(self, subtokens, maxlen=-1):
        unpadded_left = [self.bert_tokenizer.vocab[st] for st in subtokens[:-1]]
        if maxlen > 0:
            unpadded_left = unpadded_left[:maxlen]
            unpadded_left = unpadded_left + \
                [0] * (maxlen - len(unpadded_left) - 1) + \
                [self.bert_tokenizer.vocab['[SEP]']]
        return torch.LongTensor(unpadded_left)

    def forward(self, src_tokens_str: List[List[str]], src_tokens=None, batch_major=True, padding_str='<pad>',
                **kwargs):

        src_tokens_str = [[t for t in seq if t != '<pad>'] for seq in src_tokens_str]
        subtoken_str = []
        merge_to_previous = []
        for seq in src_tokens_str:
            ss, mm = self._subtokenize_sequence(seq)
            subtoken_str.append(ss)
            merge_to_previous.append(mm)

        max_token_len = max(map(len, src_tokens_str))
        max_subtoken_len = max(map(len, subtoken_str))

        contextual_embeddings = torch.zeros(len(src_tokens_str), max_token_len, self.embedding_dim, dtype=torch.float32).to(self.device)
        input_ids = torch.stack([self._convert_to_indices(seq, max_subtoken_len) for seq in subtoken_str], dim=0).to(self.device)

        attention_mask = input_ids > 0
        token_type_ids = torch.zeros_like(input_ids, dtype=torch.int64)

        raw_output, _ = \
            self.bert_model.forward(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, output_all_encoded_layers=False)
        raw_output = raw_output[:,1:]
        for n_seq, merge_seq in enumerate(merge_to_previous):

            n_token = 0
            n_subtoken_per_token = 1
            for n_subtoken, merge in enumerate(merge_seq[1:]):
                prev = contextual_embeddings[n_seq, n_token]
                next = raw_output[n_seq, n_token]
                contextual_embeddings[n_seq, n_token] = prev + (next - prev) / n_subtoken_per_token
                if merge:
                    n_subtoken_per_token += 1
                else:
                    n_subtoken_per_token = 1
                    n_token += 1

        if not batch_major:
            contextual_embeddings = contextual_embeddings.transpose(0, 1)

        return {'inner_states': [contextual_embeddings]}

    @property
    def embedding_dim(self):
        return self.bert_model.encoder.layer[-1].output.dense.out_features

    @property
    def n_hidden_states(self):
        return 1

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def is_cuda(self):
        return next(self.parameters()).is_cuda



class QBERTEmbedder(BaseContextualEmbedder):
    """Requirements for a class to be used as a contextual embedder:
        1) must have a compute_hidden_states method which returns a dict with at least:
            (see the signature in forward)
            - "embedded" the embedded sequence
            - "inner_states" all the hidden representations
        2) must have a n_hidden_states attribute""
        3) must have a embed_dim attribute
    """

    BI_TRANSFORMER = BiTransformerLanguageModel.build_model_decoder

    def __init__(self, checkpoint_or_name: str, build_fn=None, extra_args={}, use_all_hidden: bool = False, use_embeddings: bool = True,
                 normalize_embeddings: bool = False, retrain_model: bool = False, padding_idx=1):
        super().__init__(retrain_model=retrain_model)

        assert not retrain_model

        if build_fn is None:
            build_fn = self.LOADABLE_CLASSES.default()

        self.load_checkpoint(build_fn=build_fn, extra_args=extra_args, checkpoint_or_name=checkpoint_or_name)

        self.use_all_hidden = use_all_hidden
        self.use_embeddings = use_embeddings
        self.normalize_embeddings = normalize_embeddings

        self.embedding_dim = self.decoder.embedding_dim
        self.padding_idx = padding_idx

        self.n_hidden_states = len(self.decoder.layers) + self.decoder.use_biattention

        for par in self.parameters():
            par.requires_grad = False

    @classmethod
    def from_args(cls, args, extra_args={}, **kwargs):
        checkpoint_args = torch.load(args.context_embeddings_qbert_checkpoint, map_location="cpu")["args"]
        if "transformer_bilm" in checkpoint_args.arch:
            build_fn = cls.BI_TRANSFORMER
            assert "dictionary" in extra_args
        else:
            raise NotImplementedError
        return cls(
            args.context_embeddings_qbert_checkpoint,
            build_fn=build_fn,
            extra_args=extra_args,
            use_all_hidden=args.context_embeddings_use_all_hidden,
            retrain_model=args.context_embeddings_trainable,
            use_embeddings=args.context_embeddings_use_embeddings,
            normalize_embeddings=args.context_embeddings_normalize_embeddings,

            **kwargs)


    def load_checkpoint(self, checkpoint_or_name, build_fn, extra_args, map_location='cpu'):
        data = torch.load(checkpoint_or_name, map_location=map_location)
        self.decoder = build_fn(args=data['args'], **extra_args)
        self.load_state_dict(data['model'], strict=False)

    def _get_logits_from_adaptive_softmax(self, input):
        adaptive_softmax = self.decoder.adaptive_softmax
        bsz, length, dim = input.size()
        input = input.contiguous().view(-1, dim)
        head_y = adaptive_softmax.head(input)
        log_probs = head_y.new_zeros(input.size(0), adaptive_softmax.vocab_size)

        head_sz = adaptive_softmax.cutoff[0] + len(adaptive_softmax.tail)
        log_probs[:, :head_sz] = head_y
        tail_priors = log_probs[:, adaptive_softmax.cutoff[0] - adaptive_softmax.buggy_offset: head_sz - adaptive_softmax.buggy_offset].clone()

        for i in range(len(adaptive_softmax.tail)):
            start = adaptive_softmax.cutoff[i]
            end = adaptive_softmax.cutoff[i + 1]
            tail_out = log_probs[:, start:end]
            tail_out.copy_(adaptive_softmax.tail[i](input))
            log_probs[:, start:end] = tail_out

        log_probs = log_probs.view(bsz, length, -1)
        return log_probs

    def get_log_prob(self,
            src_tokens: torch.LongTensor,
            src_tokens_str: Union[None, List[List[str]]] = None,
            batch_major: bool = True,
            **kwargs):
        out = self(
            src_tokens,
            src_tokens_str,
            batch_major, **kwargs
        )["inner_states"][-1]
        if self.decoder.adaptive_softmax is None:
            if self.decoder.share_input_output_embed:
                out = F.linear(out, self.decoder.embed_tokens.weight)
            else:
                out = F.linear(out, self.decoder.embed_out)
            return F.log_softmax(out, -1)
        else:
            return self.decoder.adaptive_softmax.get_log_prob(out, None)

    def get_logits(self,
            src_tokens: torch.LongTensor,
            src_tokens_str: Union[None, List[List[str]]] = None,
            batch_major: bool = True,
            **kwargs):
        out = self(
            src_tokens,
            src_tokens_str,
            batch_major, **kwargs
        )["inner_states"][-1]
        if self.decoder.adaptive_softmax is None:
            if self.decoder.share_input_output_embed:
                out = F.linear(out, self.decoder.embed_tokens.weight)
            else:
                out = F.linear(out, self.decoder.embed_out)
            return out
        else:
            return self._get_logits_from_adaptive_softmax(out)


    def forward(
            self,
            src_tokens: torch.LongTensor,
            src_tokens_str: Union[None, List[List[str]]] = None,
            batch_major: bool = True,
            **kwargs
    ):

        assert src_tokens is not None

        with torch.set_grad_enabled(self.retrain_model):
            out = self.decoder.compute_hidden_states(src_tokens, batch_major=True)
        return out
