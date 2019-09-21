import os
import pickle

import numpy
import torch
from argparse import ArgumentParser

from fairseq.data import Dictionary
from torch.utils.data import DataLoader

from qbert.fairseq_ext.data.dictionaries import ResourceManager, SequenceLabelingTaskKind, TargetManager, MFSManager
from qbert.fairseq_ext.data.wsd_dataset import WSDDatasetBuilder, WSDDataset
from qbert.fairseq_ext.tasks.sequence_tagging import SequenceLabelingTask

def main(args):
    
    print("Loading checkpoints: " + " ".join(args.checkpoints))

    context_window = args.context_window
    delta = args.sliding_window_evaluation
    if delta != 0:
        assert context_window % (2 * delta) == 0
        half = context_window // 2

    #os.mkdir(args.results_dir)

    data = torch.load(args.checkpoints[0], map_location='cpu')
    model_args = data['args']
    state = data['model']
    if args.embeddings_checkpoint:
        model_args.context_embeddings_qbert_checkpoint = args.embeddings_checkpoint

    if args.mfs_path:
        mfs = MFSManager.load(args.mfs_path)
        assert mfs.use_synsets == args.use_synsets
        mfs.reduce_mask_to_seen = args.only_seen_lemma_pos
    else:
        mfs = None

    target_manager = TargetManager(SequenceLabelingTaskKind.WSD, mfs=mfs)
    dictionary = Dictionary.load(args.dictionary)
    output_dictionary = ResourceManager.get_offsets_dictionary() if args.use_synsets \
        else ResourceManager.get_sensekeys_dictionary()

    task = SequenceLabelingTask(model_args, dictionary, output_dictionary)
    model = task.build_model(model_args).cpu().eval()

    if len(args.checkpoints )> 1:
        for path in args.checkpoints[1:]:
            new_state = torch.load(path, map_location='cpu')['model']
            for name, tensor in new_state.items():
                if tensor.dtype == torch.float32:
                    state[name] = state[name] + tensor
        for name, tensor in state.items():
            if tensor.dtype == torch.float32:
                state[name] = state[name] / len(args.checkpoints)


    #model.upgrade_state_dict(state)
    model.load_state_dict(state, strict=True)

    model = model.eval()
    if args.use_cuda:
        model = model.cuda()

    for corpus in args.eval_on:
        if corpus.endswith('.data.xml'):
            dataset = WSDDataset.read_raganato(
                corpus,
                dictionary,
                use_synsets=args.use_synsets,
                max_length=10000 if args.sliding_window_evaluation else context_window,
            )
        else:
            with open(corpus, 'rb') as pkl:
                dataset = pickle.load(pkl)
        hit, tot = 0, 0
        all_answers = {}
        for sample_original in DataLoader(dataset, collate_fn=dataset.collater, batch_size=1):
            if args.sliding_window_evaluation:

                ntokens = sample_original['ntokens']
                lprobs_full_sample = []
                del sample_original['id']
                del sample_original['ntokens']
                del sample_original['net_input']['src_lengths']
                for i in range(0, ntokens, delta):
                    start = i
                    end = i + context_window

                    sample = target_manager.slice_batch(sample_original, slice(start, end))
                    sample['net_input']['src_lengths'] = torch.tensor([sample['net_input']['src_tokens'].size(1)])
                    with torch.no_grad():
                        net_output = model(**{k: v.cuda() if isinstance(v, torch.Tensor) and args.use_cuda else v
                                              for k, v in sample['net_input'].items()})
                        lprobs_section = model.get_normalized_probs(net_output, log_probs=True)
                    if end >= ntokens:
                        if start >= 0:
                            lprobs_section = lprobs_section[:,half-delta:]
                        else:
                            pass
                        lprobs_full_sample.append(lprobs_section.cpu())
                        break
                    elif start == 0:
                        lprobs_section = lprobs_section[:,:half]
                    else:
                        lprobs_section = lprobs_section[:,half-delta:half]
                    lprobs_full_sample.append(lprobs_section.cpu())
                lprobs = torch.cat(lprobs_full_sample, dim=1).cpu()
                assert lprobs.size(1) == ntokens

            else:
                with torch.no_grad():
                    net_output = model(**{k: v.cuda() if isinstance(v, torch.Tensor) and args.use_cuda else v
                                              for k, v in sample_original['net_input'].items()})
                    lprobs = model.get_normalized_probs(net_output, log_probs=True).cpu()

            results, answers = target_manager.calulate_metrics(lprobs, sample_original)
            all_answers.update(answers)
            hit += results['hit']
            tot += results['tot']
        print(corpus)
        print(hit / tot)
        if args.write_results:
            all_answers = {k: output_dictionary.symbols[v] for k, v in all_answers.items()}
            if not os.path.exists(args.results_dir):
                os.mkdir(args.results_dir)
            path = os.path.join(args.results_dir, os.path.split(corpus)[-1].split('.')[0]) + '.results.key.txt'
            with open(path, 'w') as results_file:
                for k, v in sorted(all_answers.items()):
                    results_file.write(k + ' ' + v + '\n')


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('-c', '--checkpoints', type=str, nargs='+', required=True)
    parser.add_argument('-a', '--checkpoint-averaging', type=int, default=0)
    parser.add_argument('-C', '--embeddings-checkpoint', type=str, default='')
    parser.add_argument('-d', '--dictionary', type=str, required=True)
    parser.add_argument('-e', '--eval-on', type=str, required=True, nargs='+')
    parser.add_argument('-s', '--use-synsets', action='store_true')
    parser.add_argument('-r', '--write-results', action='store_true')
    parser.add_argument('-R', '--results-dir', type=str, default='results')
    parser.add_argument('-w', '--context-window', type=int, default=100)
    parser.add_argument('-W', '--sliding-window-evaluation', default=0, type=int)
    parser.add_argument('-m', '--mfs-path', type=str)
    parser.add_argument('-H', '--load-from-hdf5', action='store_true')
    parser.add_argument('--only-seen-lemma-pos', action='store_true')
    parser.add_argument('-g', '--use-cuda', action='store_true')
    args = parser.parse_args()

    args.checkpoints = checkpoints = sorted(list(set(map(os.path.realpath, args.checkpoints))), key=lambda x: os.path.getmtime(x))

    assert args.checkpoint_averaging >= 0

    if args.checkpoint_averaging == 0:
        main(args)
    else:
        for i, _ in enumerate(checkpoints):
            args.checkpoints = checkpoints[max(i-args.checkpoint_averaging+1, 0):i+1]
            main(args)



