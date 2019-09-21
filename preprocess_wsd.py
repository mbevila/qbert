import argparse
import os

from fairseq.data import Dictionary

from qbert.fairseq_ext.data.dictionaries import MFSManager, ResourceManager
from qbert.fairseq_ext.data.wsd_dataset import WSDDataset, WSDDatasetBuilder

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dictionary', required=True)
    parser.add_argument('-o', '--output', required=True)
    parser.add_argument('-x', '--xml', required=False, type=str, nargs='*')
    parser.add_argument('-p', '--plaintext', required=False, type=str, nargs='*')
    parser.add_argument('-b', '--babelfy', required=False, type=str, nargs='*')
    parser.add_argument('-s', '--synsets', required=False, action='store_true')
    parser.add_argument('-l', '--max-length', required=False, type=int, default=100)
    parser.add_argument('-m', '--calc-mfs', required=False, action='store_true')
    parser.add_argument('-k', '--keep-oov', required=False, action='store_true')
    args = parser.parse_args()

    dictionary = Dictionary.load(args.dictionary)
    output_dictionary = ResourceManager.get_senses_dictionary(args.synsets)

    output = WSDDatasetBuilder(args.output, dictionary=dictionary, use_synsets=args.synsets, keep_string_data=args.keep_oov)

    if args.xml:
        for xml_path in args.xml:
            output.add_raganato(xml_path=xml_path, max_length=args.max_length)

    if args.plaintext:
        for plaintext_path in args.plaintext:
            raise NotImplementedError

    if args.babelfy:
        for babelfy_path in args.babelfy:
            raise NotImplementedError

    output.finalize()

    dataset = WSDDataset(args.output, dictionary=dictionary, use_synsets=args.synsets)

    if args.calc_mfs:
        mfs = MFSManager.from_datasets([dataset], use_synsets=args.synsets)
        mfs.save(os.path.join(args.output, 'mfs.txt'))
