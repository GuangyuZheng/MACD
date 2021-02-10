from __future__ import division
from __future__ import print_function

import numpy as np
from easydict import EasyDict as edict

__C = edict()
cfg = __C

__C.TASK_NAME = 'flickr'
__C.IMAGE_DIR = '/home/wanyuncui/cv_data/flicker30k/flickr30k-images'
__C.DATA_DIR = '/home/wanyuncui/vlm/datasets/flickr/'
__C.OUTPUT_DIR = '/home/wanyuncui/vlm/vlm-pretrained/'
__C.NUM_WORKERS = 4

__C.GPU_ID = 0

__C.TRAIN = True
__C.EVAL = True
__C.BATCH_SIZE = 128
__C.GRAD_ACCUM = 4
__C.EPOCH = 20
__C.LR = 5e-5

__C.LWF = edict()
__C.LWF.MODEL_TYPE = 'bert'
__C.LWF.MODEL_NAME = 'bert-base-uncased'
__C.LWF.CACHE_DIR = '/home/wanyuncui/nlp_data/bert/'
__C.LWF.MAX_LEN = -1
__C.LWF.LOWER_CASE = True
__C.LWF.LAMBDA0 = 1.0
__C.LWF.FREEZE_POS_EMBS = True
__C.LWF.ALPHA_CE = 5.0
__C.LWF.TEMPERATURE = 2.0
__C.LWF.ALPHA_COS = 1.0
__C.LWF.LAYER_WISE = False

__C.TEXT = edict()
__C.TEXT.MODEL_TYPE = 'bert'
__C.TEXT.MODEL_NAME = 'bert-base-uncased'
__C.TEXT.CACHE_DIR = '/home/wanyuncui/nlp_data/bert/'
__C.TEXT.MAX_LEN = 64
__C.TEXT.LOWER_CASE = True

__C.IMAGE = edict()
__C.IMAGE.ENCODER = 'resnet-50'
__C.IMAGE.RESIZE_SIZE = 224
__C.IMAGE.CROP_SIZE = 224
__C.IMAGE.MODEL_PATH = ''

__C.SMOOTH = edict()
__C.SMOOTH.TYPE = 'unconditioned'
__C.SMOOTH.GAMMA1 = 1.0
__C.SMOOTH.GAMMA2 = 5.0
__C.SMOOTH.GAMMA3 = 10.0
__C.SMOOTH.LAMBDA = 1.0

__C.MACD = edict()
__C.MACD.LOCAL_WEIGHT = 1.0
__C.MACD.GLOBAL_WEIGHT = 1.0


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
