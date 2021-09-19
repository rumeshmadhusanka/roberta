from typing import Any, Dict, Optional
from overrides import overrides
from allennlp.training.metrics.metric import Metric
import torch
from allennlp.nn.util import dist_reduce_sum
import pdb

class Self_Accuracy(Metric):
    """
    A class representing a metric of accuracy for RACE-h outputted by T5
    """
    def __init__(self) -> None:
        self._correct_count = 0.0
        self._total_count = 0

    @overrides
    def __call__(
        self, predictions: torch.Tensor, gold_labels: torch.Tensor, mask: Optional[torch.BoolTensor] = None,
    ):
        """
        # Parameters
        predictions : `torch.Tensor`, required.
            A tensor of predictions.
        gold_labels : `torch.Tensor`, required.
            A tensor corresponding to some gold label to evaluate against.
        mask : `torch.BoolTensor`, optional (default = `None`).
            A mask can be passed, in order to deal with metrics which are
            computed over potentially padded elements, such as sequence labels.
        """
        predictions, gold_labels, mask = self.detach_tensors(predictions, gold_labels, mask)

        if mask is not None and mask.size() != predictions.size():
            raise ValueError(
                f"mask must have shape == predictions.size() but "
                f"found tensor of shape: {mask.size()}"
            )

        batch_size = predictions.size(0)

        if mask is not None:
            # We can multiply by the mask up front, because we're just checking equality below, and
            # this way everything that's masked will be equal.
            predictions = predictions * mask
            gold_labels = gold_labels * mask

            # We want to skip predictions that are completely masked;
            # so we'll keep predictions that aren't.
            keep = mask.view(batch_size, -1).max(dim=1)[0]
        else:
            keep = torch.ones(batch_size, device=predictions.device).bool()
        
        predictions = predictions.view(batch_size, -1)
        gold_labels = gold_labels.view(batch_size, -1)
        if predictions.shape[-1] > gold_labels.shape[-1]:
            special_token_index = (gold_labels == 1).nonzero()
            correct = 0
            _total_count = torch.tensor(0, device='cuda')
            for i in range(gold_labels.shape[0]):
                if (special_token_index[:, 0] == i).sum() != 0:
                    # pdb.set_trace()
                    special_index = (special_token_index[:, 0] == i).nonzero()[0][0]
                    correct += predictions[i, :special_token_index[special_index, 1]].eq(gold_labels[i, :special_token_index[special_index, 1]]).prod(dim=0).float()
                else:
                    correct += predictions[i, :gold_labels.shape[-1]].eq(gold_labels[i, :]).prod(dim=0).float()
                # _total_count += (gold_labels == 1).nonzero().sum(dim=0)[1]-gold_labels.shape[0]
                _total_count += 1
        else:
            # correct = predictions.eq(gold_labels[:, :predictions.shape[-1]]).sum(dim=1).float()
            # _total_count = (predictions == 1).nonzero().sum(dim=0)[1]-predictions.shape[0]
            special_token_index = (predictions == 1).nonzero()
            correct = 0
            _total_count = torch.tensor(0, device='cuda')
            for i in range(predictions.shape[0]):
                if (special_token_index[:, 0] == i).sum() != 0:
                    special_index = (special_token_index[:, 0] == i).nonzero()[0][0]
                    correct += gold_labels[i, :special_token_index[special_index, 1]].eq(predictions[i, :special_token_index[special_index, 1]]).prod(dim=0).float()
                else:
                    correct += predictions[i, :].eq(gold_labels[i, :predictions.shape[-1]]).prod(dim=0).float()
                _total_count += 1


        # _correct_count = (correct * keep).sum()
        self._correct_count += dist_reduce_sum(correct).item()
        self._total_count += dist_reduce_sum(_total_count)

    @overrides
    def get_metric(self, reset: bool) -> Dict[str, Any]:
        """
        Compute and return the metric. Optionally also call `self.reset`.
        """
        if self._total_count > 0:
            accuracy = float(self._correct_count) / float(self._total_count)
        else:
            accuracy = 0.0
        if reset:
            self.reset()
        return accuracy

    @overrides
    def reset(self) -> None:
        """
        Reset any accumulators or internal state.
        """
        self._correct_count = 0.0
        self._total_count = 0.0