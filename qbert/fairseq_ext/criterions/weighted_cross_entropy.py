import math

import torch
from torch.nn import functional as F

from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq import utils

from qbert.fairseq_ext.data.dictionaries import TargetManager

@register_criterion('weighted_cross_entropy')
class WeightedCrossEntropyCriterionWithSoftmaxMask(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        if hasattr(task, "criterion_weights") and task.criterion_weights is not None:
            self.weight = torch.nn.Parameter(task.criterion_weights)
            self.weight.requires_grad = False
        else:
            self.weight = None

        self.trg_manager = TargetManager(args.kind)

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """


        net_output = model(**sample['net_input'])
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        results, _ = self.trg_manager.calulate_metrics(lprobs, sample)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = self.trg_manager.get_targets(sample).view(-1)
        #target = model.get_targets(sample, net_output).view(-1)

        loss = F.nll_loss(
            lprobs,
            target,
            weight=self.weight,
            size_average=False,
            ignore_index=self.padding_idx,
            reduce=reduce
        )
        sample_size = self.trg_manager.get_targets(sample).size(0) if self.args.sentence_avg else sample['ntokens']
        logging_output = {
            'hit': results['hit'],
            'tot': results['tot'],
            'loss': utils.item(loss.data) if reduce else loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': self.trg_manager.get_targets(sample).size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        try:
            acc = sum(log.get('hit') for log in logging_outputs) / sum(log.get('tot') for log in logging_outputs)
        except ZeroDivisionError:
            acc = 0.
        agg_output = {
            #'acc': acc,
            'hit': sum(log.get('hit') for log in logging_outputs),
            'tot': sum(log.get('tot') for log in logging_outputs),
            'loss': loss_sum / sample_size / math.log(2),
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }
        if sample_size != ntokens:
            agg_output['nll_loss'] = loss_sum / ntokens / math.log(2)
        return agg_output
