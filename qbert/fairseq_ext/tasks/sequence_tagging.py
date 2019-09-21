import itertools
import pathlib
import pickle

import numpy as np
import torch

from fairseq.data import Dictionary
from fairseq.tasks import FairseqTask, register_task
import os

from qbert.fairseq_ext.data.dictionaries import ResourceManager
from qbert.fairseq_ext.data.wsd_dataset import WSDDataset
from torch.utils.data import ConcatDataset

@register_task('sequence_tagging')
class SequenceLabelingTask(FairseqTask):

    source_dictionary: Dictionary
    target_dictionary: Dictionary
    kind: str
    """
    Train a language model.

    Args:
        dictionary (~fairseq_ext.data.Dictionary): the dictionary for the input of
            the language model
        output_dictionary (~fairseq_ext.data.Dictionary): the dictionary for the
            output of the language model. In most cases it will be the same as
            *dictionary*, but could possibly be a more limited version of the
            dictionary (if ``--output-dictionary-size`` is used).
        targets (List[str]): list of the target types that the language model
            should predict.  Can be one of "self", "future", and "past".
            Defaults to "future".

    .. note::

        The language modeling task is compatible with :mod:`train.py <train>`,
        :mod:`generate.py <generate>`, :mod:`interactive.py <interactive>` and
        :mod:`eval_lm.py <eval_lm>`.

    The language modeling task provides the following additional command-line
    arguments:

    .. argparse::
        :ref: fairseq_ext.tasks.language_modeling_parser
        :prog:
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('data', help='path to data directory')
        parser.add_argument('--kind', default='wsd', type=str, help='kind of labels to train and evaluate on')
        parser.add_argument('--tokens-per-sample', default=1024, type=int,
                            help='max number of tokens per sample for LM dataset')
        parser.add_argument('--raw-text', default=False, action='store_true',
                            help='load raw text dataset')
        parser.add_argument('--output-dictionary-size', default=-1, type=int,
                            help='limit the size of output dictionary')
        # fmt: on

    def __init__(self, args, dictionary: Dictionary, output_dictionary: Dictionary, criterion_weights=None):
        super().__init__(args)

        assert isinstance(dictionary, Dictionary)
        assert isinstance(output_dictionary, Dictionary)

        self.dictionary = dictionary
        self.output_dictionary = output_dictionary
        self.kind = args.kind
        self.criterion_weights = criterion_weights

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        dictionary = None
        output_dictionary = None
        if args.data:
            dictionary = Dictionary.load(os.path.join(args.data, 'dict.txt'))
            print('| dictionary: {} types'.format(len(dictionary)))
            output_dictionary = ResourceManager.get_senses_dictionary(True)
            print('| output_dictionary: {} types'.format(len(output_dictionary)))
            criterion_weights = torch.ones(len(output_dictionary)).float()
            criterion_weights[:output_dictionary.nspecial] = 0.
            criterion_weights.requires_grad = False
        else:
            raise NotImplementedError

        return cls(args, dictionary, output_dictionary, criterion_weights=criterion_weights)

    def build_model(self, args):
        model = super().build_model(args)
        return model


    def load_dataset(self, split, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """

        loaded_datasets = []

        for k in itertools.count():
            split_k = split + (str(k) if k > 0 else '')
            path = os.path.join(self.args.data, split_k)
            path = str(pathlib.Path(path).resolve())
            print(path)
            if not os.path.exists(path):
                if k > 0:
                    break
                else:
                    raise FileNotFoundError('Dataset not found: {} ({})'.format(split, self.args.data))

            dataset = WSDDataset(path, self.dictionary, True, add_monosemous=split == 'train')
            dataset.load_in_memory()
            loaded_datasets.append(dataset)

            if not combine:
                break

        if len(loaded_datasets) == 1:
            dataset = loaded_datasets[0]
        else:
            dataset = ConcatDataset(loaded_datasets)

        self.datasets[split] = dataset

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq_ext.data.Dictionary` for the sequence
        labeling model."""
        return self.dictionary

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq_ext.data.Dictionary` for the sequence
        labeling model."""
        return self.output_dictionary