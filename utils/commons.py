import json
import numpy
import torch
import torch.nn.functional as F


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.integer):
            return int(obj)
        elif isinstance(obj, numpy.floating):
            return float(obj)
        elif isinstance(obj, numpy.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


def transform_softmax(x, dim):
    m, _ = torch.max(x, dim=dim, keepdim=True)
    x = x - m
    return F.softmax(x, dim=dim)


def modified_cross_entropy(inputs, target, mask, t=2, reduction='sum'):
    mask = mask.expand(inputs.size()).float()

    inputs = inputs / t
    target = target / t

    target_prob = F.softmax(target, dim=1)
    log_inputs_prob = -F.log_softmax(inputs, dim=1)

    loss = torch.mul(target_prob, log_inputs_prob) * mask
    if reduction == 'average':
        loss = torch.sum(loss, keepdim=True, dim=-1).mean()
    else:
        loss = torch.sum(loss)
    return loss


# def modified_cross_entropy(inputs, target, mask, t=2, reduction='sum'):
#     mask = mask.float()
#
#     inputs_prob = F.softmax(inputs, dim=1)
#     target_prob = F.softmax(target, dim=1)
#
#     inputs_prob = inputs_prob ** (1/t)
#     target_prob = target_prob ** (1/t)
#
#     inputs_prob = inputs_prob / torch.sum(inputs_prob, dim=1, keepdim=True)
#     target_prob = target_prob / torch.sum(target_prob, dim=1, keepdim=True)
#
#     log_inputs_prob = -torch.log(inputs_prob)
#
#     batch = inputs.shape[0]
#     loss = torch.mul(target_prob, log_inputs_prob) * mask.view(batch, -1).expand(inputs.size())
#     if reduction == 'average':
#         loss = torch.sum(loss) / batch
#     else:
#         loss = torch.sum(loss)
#     return loss
