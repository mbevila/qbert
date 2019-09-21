import enum
import os
import pickle
from collections import defaultdict, OrderedDict
from typing import Set, List, Union

import torch
from fairseq.data import Dictionary
from nltk.corpus import wordnet
import numpy as np

from qbert.utils import QBERT_RES_DIR
from qbert.fairseq_ext.data.utils import make_offset


#from qbert.fairseq_ext.data.wsd_dataset import WSDDataset


class MFSManager:

    _INVALID_INTS = {0, 1, 2, 3}

    def __init__(self, mfs_dict, use_synsets=False, reduce_mask_to_seen=False):
        if use_synsets:
            output_dictionary = ResourceManager.get_offsets_dictionary()
            lemma_pos_to_possible = ResourceManager.get_lemma_pos_to_possible_offsets()
        else:
            output_dictionary = ResourceManager.get_sensekeys_dictionary()
            lemma_pos_to_possible = ResourceManager.get_lemma_pos_to_possible_sensekeys()
        self.lemma_pos_dictionary = ResourceManager.get_lemma_pos_dictionary()
        self.lemma_pos_to_possible = lemma_pos_to_possible
        self.output_dictionary = output_dictionary
        self.mfs_dict = mfs_dict
        self.use_synsets = use_synsets
        self.reduce_mask_to_seen = reduce_mask_to_seen

    def get_mask_indices(self, lemma_pos):
        if lemma_pos not in self.mfs_dict:
            return self.lemma_pos_to_possible[lemma_pos][0:1]
        elif self.reduce_mask_to_seen:
            poss = self.mfs_dict[lemma_pos]
            poss = [p[0] for p in poss]
            return poss
        else:
            return self.lemma_pos_to_possible[lemma_pos]

    def save(self, path):
        with open(path, 'w') as export:
            export.write(f'# use_synsets:\t{str(int(self.use_synsets))}\n')
            export.write(f'# reduce_mask_to_seen:\t{str(int(self.reduce_mask_to_seen))}\n')
            for lemma_pos, senses_count_couples in self.mfs_dict.items():
                export.write(self.lemma_pos_dictionary.symbols[lemma_pos]+'\t')
                export.write('\t'.join([self.output_dictionary.symbols[sense] + '@@@' + str(cnt) for sense, cnt in senses_count_couples]))
                export.write('\n')

    @classmethod
    def load(cls, path):
        mfs_dict = {}
        with open(path, 'r') as import_:
            use_synsets = bool(int(next(import_).strip().split('\t')[-1]))
            reduce_mask_to_seen = bool(int(next(import_).strip().split('\t')[-1]))
            inst = cls(mfs_dict, use_synsets=use_synsets, reduce_mask_to_seen=reduce_mask_to_seen)
            for line in import_:
                line = line.strip()
                if not line: continue
                chunks = line.split('\t')
                lp = inst.lemma_pos_dictionary.index(chunks[0])
                mfs_dict[lp] = \
                    [(inst.output_dictionary.index(chnk.split('@@@')[0]), int(chnk.split('@@@')[1])) for chnk in chunks[1:]]
        return inst

    @staticmethod
    def _open_dataset_pickle(dataset_pickle_path: str):# -> WSDDataset:
        with open(dataset_pickle_path, 'rb') as bin:
            dataset = pickle.load(bin)
            #assert isinstance(dataset, WSDDataset)
        return dataset

    @classmethod
    def from_datasets(cls, datasets, use_synsets=False, reduce_mask_to_seen=False):
        raise NotImplementedError
        counts = defaultdict(dict)
        mfs_dict = OrderedDict()
        for dataset in datasets:
            has_gold_mask = [len(cls._INVALID_INTS.intersection(set(x[1]))) == 0 for x in dataset.gold]
            gold = [g for g, has_gold in zip(dataset.gold, has_gold_mask) if has_gold]
            lemma_pos = dataset.data[has_gold_mask].lemma_pos
            for g, lp in zip(gold, lemma_pos):
                for g_idx in g:
                    counts[lp][g_idx] = counts[lp].get(g_idx, 0) + 1
        for lp_idx, gold_counts in sorted(counts.items(), key=lambda kv: sum(kv[1].values()), reverse=True):
            mfs_dict[lp_idx] = sorted(gold_counts.items(), key=lambda kv: kv[1], reverse=True)
        return cls(mfs_dict, use_synsets=use_synsets, reduce_mask_to_seen=reduce_mask_to_seen)


    @classmethod
    def from_pickles(cls, dataset_pickle_paths: List[str], use_synsets=False, reduce_mask_to_seen=False):
        datsets = (cls._open_dataset_pickle(path) for path in dataset_pickle_paths)
        return cls.from_datasets(datasets=datsets, use_synsets=use_synsets, reduce_mask_to_seen=reduce_mask_to_seen)

class ResourceManager:

    _pos_dictionary = None
    _sensekeys_dictionary = None
    _offsets_dictionary = None
    _lemma_pos_dictionary = None
    _lemma_pos_to_possible_sensekeys = None
    _lemma_pos_to_possible_offsets = None
    _sensekeys_weights = None
    _offsets_weights = None

    @classmethod
    def get_pos_dictionary(cls) -> Dictionary:
        if cls._pos_dictionary is None:
            cls._pos_dictionary = Dictionary.load(os.path.join(QBERT_RES_DIR, 'dictionaries/pos.txt'))
        return cls._pos_dictionary

    @classmethod
    def get_sensekeys_dictionary(cls) -> Dictionary:
        if cls._sensekeys_dictionary is None:
            cls._sensekeys_dictionary = Dictionary.load(os.path.join(QBERT_RES_DIR, 'dictionaries/sensekeys.txt'))
        return cls._sensekeys_dictionary

    @classmethod
    def get_offsets_dictionary(cls) -> Dictionary:
        if cls._offsets_dictionary is None:
            cls._offsets_dictionary = Dictionary.load(os.path.join(QBERT_RES_DIR, 'dictionaries/offsets.txt'))
        return cls._offsets_dictionary

    @classmethod
    def get_senses_dictionary(cls, use_synsets=False):
        if use_synsets:
            return cls.get_offsets_dictionary()
        else:
            return cls.get_sensekeys_dictionary()

    @classmethod
    def get_lemma_pos_dictionary(cls) -> Dictionary:
        if cls._lemma_pos_dictionary is None:
            cls._lemma_pos_dictionary = Dictionary.load(os.path.join(QBERT_RES_DIR, 'dictionaries/lemma_pos.txt'))
        return cls._lemma_pos_dictionary

    @classmethod
    def get_lemma_pos_to_possible_sensekeys(cls) -> List[Set[int]]:
        if cls._lemma_pos_to_possible_sensekeys is None:
            lemma_pos_dictionary = cls.get_lemma_pos_dictionary()
            sensekeys_dictionary = cls.get_sensekeys_dictionary()
            lemma_pos_to_possible_sensekeys = []
            for i, lemma_pos in enumerate(lemma_pos_dictionary.symbols):
                if i < lemma_pos_dictionary.nspecial:
                    lemma_pos_to_possible_sensekeys.append([lemma_pos_dictionary.index(lemma_pos)])
                else:
                    lemma, pos = lemma_pos.split('#')
                    senses = [sensekeys_dictionary.index(l.key()) for l in wordnet.lemmas(lemma, pos)]
                    lemma_pos_to_possible_sensekeys.append(senses)
            cls._lemma_pos_to_possible_sensekeys = lemma_pos_to_possible_sensekeys
        return cls._lemma_pos_to_possible_sensekeys

    @classmethod
    def get_lemma_pos_to_possible_offsets(cls) -> List[Set[int]]:
        if cls._lemma_pos_to_possible_offsets is None:
            lemma_pos_dictionary = cls.get_lemma_pos_dictionary()
            offsets_dictionary = cls.get_offsets_dictionary()
            lemma_pos_to_possible_offsets = []
            for i, lemma_pos in enumerate(lemma_pos_dictionary.symbols):
                if i < lemma_pos_dictionary.nspecial:
                    lemma_pos_to_possible_offsets.append([lemma_pos_dictionary.index(lemma_pos)])
                else:
                    lemma, pos = lemma_pos.split('#')
                    senses = [offsets_dictionary.index(make_offset(s)) for s in wordnet.synsets(lemma, pos)]
                    lemma_pos_to_possible_offsets.append(senses)
            cls._lemma_pos_to_possible_offsets = lemma_pos_to_possible_offsets
        return cls._lemma_pos_to_possible_offsets

    @classmethod
    def get_lemma_pos_to_possible_senses(cls, use_synsets=False) -> List[Set[int]]:
        if use_synsets:
            return cls.get_lemma_pos_to_possible_offsets()
        else:
            return cls.get_lemma_pos_to_possible_sensekeys()

    @classmethod
    def get_sensekey_weights(cls):
        if cls._sensekeys_weights is None:
            weights = []
            for s in cls.get_sensekeys_dictionary().symbols:
                if s.startswith("<"):
                    weights.append(0.0)
                else:
                    weights.append(1.0)
            cls._sensekeys_weights = np.array(weights)
        return cls

class SequenceLabelingTaskKind(enum.Enum):

    WSD = 0
    POS = 1

class TargetManager:

    def __init__(self, kind: Union[str, SequenceLabelingTaskKind], **other_stuff):
        if isinstance(kind, str):
            self.kind = SequenceLabelingTaskKind[kind.upper()]
        else:
            self.kind = kind
        assert isinstance(self.kind, SequenceLabelingTaskKind)
        self._setup(other_stuff)

    def _setup(self, other_stuff):
        getattr(self, '_setup_' + self.kind.name, lambda other_stuff: None)(other_stuff)

    def calulate_metrics(self, lprobs, sample):
        return getattr(self, '_calculate_metrics_' + self.kind.name, self._calculate_metrics_DEFAULT)(lprobs, sample)

    def get_targets(self, sample):
        return getattr(self, '_get_targets_' + self.kind.name)(sample)

    def _setup_WSD(self, other_stuff):
        if other_stuff.get('mfs'):
            assert isinstance(other_stuff['mfs'], MFSManager)
            self.mfs_manager = other_stuff['mfs']
        else:
            self.mfs_manager = None
        self.only_calc_on_ids = other_stuff.get('only_calc_on_ids', True)

    def _calculate_metrics_DEFAULT(self, lprobs, sample):
        preds = lprobs.view(-1, lprobs.size(-1)).argmax(1)
        true = self.get_targets(sample).view(-1)
        return {'hit': (preds == true).sum().item(), 'tot': len(true)}, preds.detach()

    def _calculate_metrics_WSD(self, lprobs, sample):
        senses = sample['senses']
        answers = {}
        hit = 0
        tot = 0
        for i, (ids, all_, gold, lemma_pos) in enumerate(zip(senses['ids'], senses['all'], senses['gold'], sample['lemma_pos'])):
            if self.only_calc_on_ids:
                js = [j for j, id_ in enumerate(ids) if id_]
            else:
                js = list(range(len(ids)))
            for j in js:
                if self.mfs_manager:
                    poss = self.mfs_manager.get_mask_indices(lemma_pos[j].item())
                else:
                    poss = list(all_[j])
                try:
                    pred = poss[lprobs[i][j][poss].argmax().item()]
                except TypeError as e:
                    print(all_)
                    print(ids[j])
                    print(gold[j])
                    print(poss)
                    print(i)
                    print(j)
                    raise e
                tot += 1
                if pred in gold[j]:
                    hit += 1
                if self.only_calc_on_ids:
                    answers[ids[j]] = pred
        return {'hit': hit, 'tot': tot}, answers

    def _get_targets_DEFAULT(self, sample):
        return sample['target']

    def _get_targets_WSD(self, sample):
        return sample['senses']['target']

    def _get_targets_POS(self, sample):
        return sample['pos']

    @classmethod
    def slice_batch(cls, batch, slice):
        assert isinstance(batch, dict)
        sliced = {}
        for k, v in batch.items():
            if isinstance(v, dict):
                sliced[k] = cls.slice_batch(v, slice)
            elif isinstance(v, torch.Tensor):
                sliced[k] = v[:, slice]
            elif isinstance(v, list):
                sliced[k] = [inner[slice] for inner in v]
            else:
                raise TypeError
        return sliced