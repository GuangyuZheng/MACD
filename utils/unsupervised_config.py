from __future__ import division
from __future__ import print_function

import numpy as np
from easydict import EasyDict as edict

__C = edict()
cfg = __C

__C.TASK_NAME = 'rte'
__C.DATA_DIR = '/home/a/vlm/nlp_data/RTE'
__C.OUTPUT_DIR = '/home/a/vlm/finetuned-models/rte'
__C.NUM_WORKERS = 1

__C.GPU_ID = 0

__C.TRAIN = True
__C.EVAL = True
__C.BATCH_SIZE = 16
__C.GRAD_ACCUM = 2
__C.EPOCH = 3
__C.LR = 5e-5
'''Used only in allennlp'''
__C.PATIENCE = 10
__C.METRIC = 'acc'

__C.TEXT = edict()
__C.TEXT.MODEL_TYPE = 'bert'
__C.TEXT.MODEL_NAME = 'bert-base-uncased'
__C.TEXT.CACHE_DIR = '/home/wanyuncui/nlp_data/bert/'
__C.TEXT.MAX_LEN = 64
__C.TEXT.LOWER_CASE = True
'''Used only in allennlp'''
__C.TEXT.INPUT_SIZE = 300
__C.TEXT.HIDDEN_SIZE = 500
__C.TEXT.PRETRAINED_EMBEDDINGS = '/home/a/nlp_data/glove/glove.840B.300d.lower.txt'


def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        if k not in b:
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                  'for config key: {}').format(type(b[k]),
                                                               type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v


def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f))

    _merge_a_into_b(yaml_cfg, __C)
