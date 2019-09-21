import enum
import math
import os
import pickle
import random
import tempfile

import h5py
from typing import List, Set, Tuple, Union, Dict
from xml.etree.cElementTree import parse

from fairseq.data import FairseqDataset, Dictionary
from nltk.corpus import wordnet
import numpy as np
import torch

from qbert.fairseq_ext.data.dictionaries import ResourceManager
from qbert.fairseq_ext.data.utils import _longest, make_offset


def _read_by_text(xml):
    for i, text in enumerate(xml.iter("text")):
        for sentence in text:
            for word in sentence:
                yield i, word

def _read_by_sent(xml):
    for i, sent in enumerate(xml.iter("sentence")):
        for word in sent:
            yield i, word

class RaganatoReadBy(enum.Enum):

    SENTENCE = _read_by_sent
    TEXT = _read_by_text

def _read_raganato_xml(xml_path: str, read_by: Union[str, RaganatoReadBy]=RaganatoReadBy.TEXT, dictionary=None) -> Tuple[np.ndarray, List[str], Dict[int, str]]:

    if isinstance(read_by, str):
        read_by = getattr(RaganatoReadBy, read_by.upper())
    elif isinstance(read_by, RaganatoReadBy):
        read_by = read_by.value

    assert dictionary is not None

    oov_dictionary = {}

    pos_dictionary = ResourceManager.get_pos_dictionary()
    lemma_pos_dictionary = ResourceManager.get_lemma_pos_dictionary()

    text_n = []
    token = []
    lemma_pos = []
    pos = []
    gold_indices = []
    target_labels = []
    gold_idx = 0

    for t_n, (i, word) in enumerate(read_by(parse(xml_path))):

        text_n.append(i)
        t = word.text.replace(' ', '_')
        if t not in dictionary.indices:
            oov_dictionary[t_n] = t
            t = _longest(t)
        token.append(dictionary.index(t))
        p = word.attrib["pos"]
        lp = lemma_pos_dictionary.index(word.attrib["lemma"].lower() + '#' + _ud_to_wn.get(p, 'x'))
        lemma_pos.append(lp)
        pos.append(pos_dictionary.index(p))
        idx = word.attrib.get("id")
        if idx:
            target_labels.append(idx)
            gold_indices.append(gold_idx)
            gold_idx += 1

        else:
            gold_indices.append(-1)

    text_n = np.array(text_n, dtype=np.int64)
    token = np.array(token, dtype=np.int32)
    lemma_pos = np.array(lemma_pos, dtype=np.int32)
    pos = np.array(pos, dtype=np.int8)
    gold_indices = np.array(gold_indices, dtype=np.int32)

    raw_data = np.rec.fromarrays(
        [text_n, token, lemma_pos, pos, gold_indices],
        names=['text_n', 'token', 'lemma_pos', 'pos', 'gold_indices']
    )

    return raw_data, target_labels, oov_dictionary


def _read_raganato_gold_(gold_path: str, use_synsets: bool = False) -> Dict[str, List[int]]:
    target_dict = {}
    dictionary = \
        ResourceManager.get_offsets_dictionary() if use_synsets else ResourceManager.get_sensekeys_dictionary()
    with open(gold_path, encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            instance, *sensekeys = line.split()
            if use_synsets:
                senses = [make_offset(wordnet.lemma_from_key(sk).synset()) for sk in sensekeys]
            else:
                senses = sensekeys
            target_dict[instance] = [dictionary.index(s) for s in senses]
    return target_dict

class WSDDatasetBuilder:

    def __init__(self, path, dictionary, use_synsets=True, keep_string_data=False):

        if not os.path.exists(path):
            os.makedirs(path)

        self.vectors_path = os.path.join(path, 'vectors.hdf5')
        self.gold_path = os.path.join(path, 'gold.pkl')
        if os.path.exists(self.vectors_path):
            os.remove(self.vectors_path)
        if os.path.exists(self.gold_path):
            os.remove(self.gold_path)

        self.keep_string_data = keep_string_data
        if keep_string_data:
            self.oov_dictionary_path = os.path.join(path, 'oov.pkl')
            if os.path.exists(self.oov_dictionary_path):
                os.remove(self.oov_dictionary_path)
            self.oov_dictionary = {}
        else:
            self.oov_dictionary = None

        self.hd5_file = h5py.File(self.vectors_path)
        self.token_data = None
        self.seq_data = None
        self.gold = []

        self._max_gold = 0
        self._max_sequence = 0

        self.dictionary = dictionary
        self.output_dictionary = \
            ResourceManager.get_offsets_dictionary() if use_synsets else ResourceManager.get_sensekeys_dictionary()

        self.use_synsets = use_synsets

    def __del__(self):
        if self.hd5_file is not None:
            self.hd5_file.close()

    def finalize(self):
        self.hd5_file.close()
        self.hd5_file = None
        with open(self.gold_path, 'wb') as pkl:
            pickle.dump(self.gold, pkl)
        if self.oov_dictionary is not None:
            with open(self.oov_dictionary_path, 'wb') as pkl:
                pickle.dump(self.oov_dictionary, pkl)


    def add_raganato(self, xml_path, max_length=100):
        raw_data, target_labels, oov_dictionary = _read_raganato_xml(xml_path, dictionary=self.dictionary)
        target_dict = _read_raganato_gold_(xml_path.replace('.data.xml', '.gold.key.txt'), use_synsets=self.use_synsets)
        gold = [(lab, target_dict[lab]) for lab in target_labels]
        self._add_corpus(raw_data, gold, max_length=max_length, oov_dictionary=oov_dictionary)

    def _add_corpus(self, raw_data: np.ndarray, gold: List[Tuple[str, List[int]]], max_length=100, oov_dictionary=None):

        if self.token_data is None:
            start = 0
            end = raw_data.shape[0]
            self.token_data = \
                self.hd5_file.create_dataset('token_data', shape=raw_data.shape, dtype=raw_data.dtype, maxshape=(None,))
            if oov_dictionary is not None and self.oov_dictionary is not None:
                oov_dictionary = {i + start: w for i, w in oov_dictionary.items()}
                self.oov_dictionary.update(oov_dictionary)

        else:
            start = self.token_data.shape[0]
            end = start + raw_data.shape[0]
            if oov_dictionary is not None and self.oov_dictionary is not None:
                oov_dictionary = {i + start: w for i, w in oov_dictionary.items()}
                self.oov_dictionary.update(oov_dictionary)
            self.token_data.resize(size=(self.token_data.shape[0] + raw_data.shape[0],))

        # fixing offsets
        n_gold = len(gold)
        raw_data['text_n'] += self._max_sequence
        raw_data['gold_indices'][raw_data['gold_indices'] != -1] = \
            np.arange(n_gold, dtype=raw_data['gold_indices'].dtype) + self._max_gold
        self._max_sequence = raw_data['text_n'][-1]
        self._max_gold += n_gold

        self.gold.extend(gold)

        self.token_data[start:end] = raw_data

        seq_raw_data = []
        start_seq = start
        # for _, seq in pd.Series(raw_data['text_n']).groupby(lambda x: x):
        #     seq = seq.values
        text_n = raw_data['text_n']
        for seq in np.split(text_n, np.unique(text_n, return_index=True)[1][1:]):
            n_chunks = math.ceil(len(seq) / max_length)
            for seq_chunk in np.array_split(seq, n_chunks):
                seq_raw_data.append([start_seq, len(seq_chunk)])
                start_seq += len(seq_chunk)

        seq_raw_data = np.array(seq_raw_data)
        if self.seq_data is None:
            start = 0
            end = seq_raw_data.shape[0]
            self.seq_data = \
                self.hd5_file.create_dataset('seq_data', shape=seq_raw_data.shape, dtype=seq_raw_data.dtype, maxshape=(None, 2))
        else:
            start = self.seq_data.shape[0]
            end = start + seq_raw_data.shape[0]
            self.seq_data.resize(size=(self.seq_data.shape[0] + seq_raw_data.shape[0], seq_raw_data.shape[1]))

        self.seq_data[start:end] = seq_raw_data


def _read_plaintext(path: str, spacy_model, text_split='\n'):
    with open(path) as f:
        text = f.read().strip()
    if text_split:
        text = text.split(text_split)
    else:
        text = [text]

class WSDDataset(FairseqDataset):

    #Data
    token_data: h5py.Dataset
    seq_data: h5py.Dataset
    gold: List[Set[int]]
    sizes: np.array

    #Dictionaries
    dictionary: Dictionary
    output_dictionary: Dictionary
    pos_dictionary: Dictionary
    lemma_pos_dictionary: Dictionary
    lemma_pos_to_possible_senses: List[Set[int]]

    #Parameters
    shuffle: bool
    add_monosemous: bool
    use_synsets: bool

    @classmethod
    def read_raganato(cls, xml_path, dictionary, use_synsets=True, add_monosemous=False, max_length=100):
        tmp_path = tempfile.mkdtemp('qbert-datasets')
        builder = WSDDatasetBuilder(tmp_path, dictionary, use_synsets=use_synsets, keep_string_data=True)
        builder.add_raganato(xml_path=xml_path, max_length=max_length)
        builder.finalize()
        inst = cls(tmp_path, dictionary, use_synsets=use_synsets, add_monosemous=add_monosemous)
        inst.load_in_memory()
        inst._from_tmp = True
        return inst

    def __init__(
            self,
            path: str,
            dictionary: Dictionary,
            use_synsets: bool = True,
            add_monosemous: bool = False,
            shuffle: bool = False,
    ) -> None:

        self.dictionary = dictionary
        self.output_dictionary = \
            ResourceManager.get_offsets_dictionary() \
                if use_synsets \
                else ResourceManager.get_offsets_dictionary()
        self.lemma_pos_dictionary = ResourceManager.get_lemma_pos_dictionary()
        self.lemma_pos_to_possible_senses = \
            ResourceManager.get_lemma_pos_to_possible_offsets() \
                if use_synsets \
                else ResourceManager.get_lemma_pos_to_possible_sensekeys()

        self.use_synsets = use_synsets
        self.add_monosemous = add_monosemous
        self.shuffle = shuffle

        self.path = path

        vectors_path = os.path.join(path, 'vectors.hdf5')
        gold_path = os.path.join(path, 'gold.pkl')
        oov_dictionary_path = os.path.join(path, 'oov.pkl')

        self._h5_file = h5py.File(vectors_path, mode="r")
        self._token_data_h5 = self._h5_file['.']['token_data']
        self._token_data_mem = None
        self._seq_data_h5 = self._h5_file['.']['seq_data']
        self._seq_data_mem = None
        with open(gold_path, 'rb') as pkl:
            self.gold = pickle.load(pkl)
        if os.path.exists(oov_dictionary_path):
            with open(oov_dictionary_path, 'rb') as pkl:
                self.oov_dictionary = pickle.load(pkl)
        else:
            self.oov_dictionary = None

        self.sizes = self.seq_data()[:, 1]
        self._from_tmp = False

    def token_data(self):
        if self._token_data_mem is None:
            return self._token_data_h5
        else:
            return self._token_data_mem

    def seq_data(self):
        if self._seq_data_mem is None:
            return self._seq_data_h5
        else:
            return self._seq_data_mem

    def load_in_memory(self):
        self._token_data_mem = self._token_data_h5[()]
        self._seq_data_mem = self._seq_data_h5[()]

    def clear_from_memory(self):
        self._token_data_mem = None
        self._seq_data_mem = None

    def __del__(self):
        pass
        # self._h5_file.close()
        # if self._from_tmp:
        #     shutil.rmtree(self.path)

    def __len__(self):
        return len(self.sizes)

    def __getitem__(self, item):

        # for negative indexing support
        if item < 0:
            item += len(self)

        seq_start, seq_len = self.seq_data()[item].tolist()
        data = self.token_data()[seq_start:seq_start+seq_len]

        lemma_pos = data['lemma_pos'].astype(np.int64)
        all_ = [self.lemma_pos_to_possible_senses[int(l)] for l in lemma_pos]

        gold = []
        ids = []
        target = []
        for g_idx, a in zip(data['gold_indices'], all_):
            if g_idx >= 0:
                trg_idx, g = self.gold[g_idx]
            else:
                trg_idx = None
                if self.add_monosemous and len(a) == 1:
                    g = list(a)
                else:
                    g = [self.output_dictionary.unk()]
            target.append(random.choice(g))
            gold.append(g)
            ids.append(trg_idx)

        target = torch.LongTensor(target)
        tokens_str = []
        for i, tkn in enumerate(data['token']):
            i += seq_start
            if self.oov_dictionary:
                s = self.oov_dictionary.get(i)
                if s is None:
                    s = self.dictionary.symbols[tkn]
            else:
                s = self.dictionary.symbols[tkn]
            tokens_str.append(s)

        sample = {
            'id' : item,
            'ntokens' : seq_len,
            'tokens' : torch.from_numpy(data['token'].astype(np.int64)),
            'tokens_str': tokens_str,
            'lemma_pos' : torch.from_numpy(lemma_pos),
            'senses' : {
                'target': target,
                'gold': gold,
                'ids': ids,
                'all': all_,
            }
        }
        return sample

    def __iter__(self):
        # hacky solution
        for i in range(len(self)):
            yield self[i]

    def batch_lists(self, lists, max_len=None, pad_value=None):
        if not max_len:
            max_len = max(lists, key=len)
        samples = []
        for l in lists:
            pad = [pad_value] * (max_len - len(l))
            samples.append(l + pad)
        return samples

    def batch_tensors(self, tensors, max_len=None, pad_value=1):
        if not max_len:
            max_len = max(tensors, key=lambda t: t.size(0))
        samples = []
        for t in tensors:
            pad = torch.empty(max_len-t.size(0), dtype=torch.int64).fill_(pad_value)
            samples.append(torch.cat((t, pad), dim=0))
        batch = torch.stack(samples, dim=0)
        return batch

    def collater(self, samples:List[dict]) -> dict:
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[dict]): samples to collate

        Returns:
            dict: a mini-batch with the following keys:

                - `id` (LongTensor): example IDs in the original input order
                - `ntokens` (int): total number of tokens in the batch
                - `net_input` (dict): the input to the Model, containing keys:

                  - `src_tokens` (LongTensor): a padded 2D Tensor of tokens in
                    the source sentence of shape `(bsz, src_len)`. Padding will
                    appear on the right.

                - `senses` (LongTensor): a padded 2D Tensor of tokens in the
                  target sentence of shape `(bsz, tgt_len)`. Padding will appear
                  on the right.
        """
        max_len = max(s['ntokens'] for s in samples)

        src_lengths = torch.LongTensor([s["ntokens"] for s in samples])
        src_tokens = self.batch_tensors(
            [s['tokens'] for s in samples], max_len=max_len, pad_value=self.dictionary.pad())
        lemmas = self.batch_tensors(
            [s['lemma_pos'] for s in samples], max_len=max_len, pad_value=self.lemma_pos_dictionary.pad()
        )
        senses = self.batch_tensors(
            [s['senses']['target'] for s in samples], max_len=max_len, pad_value=self.output_dictionary.pad())

        all_senses = self.batch_lists([s['senses']['all'] for s in samples], max_len=max_len, pad_value={self.output_dictionary.pad()})
        target_ids = self.batch_lists([s['senses']['ids'] for s in samples], max_len=max_len, pad_value=None)
        src_tokens_str = self.batch_lists([s['tokens_str'] for s in samples], max_len=max_len, pad_value=self.dictionary.pad_word)
        gold = self.batch_lists([s['senses']['gold'] for s in samples], max_len=max_len, pad_value={self.output_dictionary.pad()})
        batch = {
            "id": torch.LongTensor([s["id"] for s in samples]),
            "ntokens": sum(s["ntokens"] for s in samples),
            'lemma_pos': lemmas,
            "net_input": {
                "src_tokens": src_tokens,
                "src_lengths": src_lengths,
                "src_tokens_str": src_tokens_str,
            },
            "senses": {
                "target": senses,
                "gold": gold,
                "all": all_senses,
                "ids": target_ids,
            }
        }
        return batch

    def get_dummy_batch(self, num_tokens, max_positions, tgt_len=128):
        """Return a dummy batch with a given number of tokens."""
        if isinstance(max_positions, float) or isinstance(max_positions, int):
            tgt_len = min(tgt_len, max_positions)
        bsz = max(num_tokens // tgt_len, 1)
        tokens = self.dictionary.dummy_sentence(tgt_len + 2)
        lemmas = self.lemma_pos_dictionary.dummy_sentence(tgt_len + 2)
        senses = self.output_dictionary.dummy_sentence(tgt_len + 2)

        return self.collater([{
            'id': i,
            'ntokens': tokens.size(0),
            'tokens': tokens,
            'tokens_str': [self.dictionary.unk_word] * tokens.size(0),
            'lemma_pos': lemmas,
            'senses': {
                'target': senses,
                'gold': [{self.output_dictionary.unk()}] * tokens.size(0),
                'ids': [None] * tokens.size(0),
                'all': [{self.output_dictionary.unk()}] * tokens.size(0),
            }
        } for i in range(bsz)])

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return self.sizes[index]

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return self.sizes[index]

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            order = [np.random.permutation(len(self))]
        else:
            order = [np.arange(len(self))]
        order.append(self.sizes)
        return np.lexsort(order)

    @property
    def supports_prefetch(self):
        return True

    def prefetch(self, indices):
        return [self[i] for i in indices]


_ud_to_wn = {
    "NOUN": "n",
    "VERB": "v",
    "ADV": "r",
    "ADJ": "a"
}