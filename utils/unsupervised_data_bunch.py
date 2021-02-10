import logging
import copy
import json
from transformers import glue_processors, glue_output_modes
from utils.finetune_data_bunch import SNLIProcessor, StsbProcessor

logger = logging.getLogger(__name__)


class InputFeatures(object):
    """
    A single set of features of data.

    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
        token_type_ids: Segment token indices to indicate first and second portions of the inputs.
        label: Label corresponding to the input
    """

    def __init__(self, input_ids, attention_mask, token_type_ids,
                 input_ids2, attention_mask2, token_type_ids2,
                 label):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids

        self.input_ids2 = input_ids2
        self.attention_mask2 = attention_mask2
        self.token_type_ids2 = token_type_ids2

        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


def convert_examples_to_features(examples, tokenizer,
                                 max_length=512,
                                 task=None,
                                 label_list=None,
                                 output_mode=None,
                                 pad_on_left=False,
                                 pad_token=0,
                                 pad_token_segment_id=0,
                                 mask_padding_with_zero=True):
    if task is not None:
        processor = processors[task]()
        if label_list is None:
            label_list = processor.get_labels()
            logger.info("Using label list %s for task %s" % (label_list, task))
        if output_mode is None:
            output_mode = output_modes[task]
            logger.info("Using output mode %s for task %s" % (output_mode, task))

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d" % (ex_index))

        inputs = tokenizer.encode_plus(
            example.text_a,
            None,
            add_special_tokens=True,
            max_length=max_length,
        )

        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
            token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
        assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(len(attention_mask),
                                                                                            max_length)
        assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(len(token_type_ids),
                                                                                            max_length)

        inputs2 = tokenizer.encode_plus(
            example.text_b,
            None,
            add_special_tokens=True,
            max_length=max_length,
        )

        input_ids2, token_type_ids2 = inputs2["input_ids"], inputs2["token_type_ids"]

        # The mask2 has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask2 = [1 if mask_padding_with_zero else 0] * len(input_ids2)

        # Zero-pad up to the sequence length.
        padding_length = max_length - len(input_ids2)
        if pad_on_left:
            input_ids2 = ([pad_token] * padding_length) + input_ids2
            attention_mask2 = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask2
            token_type_ids2 = ([pad_token_segment_id] * padding_length) + token_type_ids2
        else:
            input_ids2 = input_ids2 + ([pad_token] * padding_length)
            attention_mask2 = attention_mask2 + ([0 if mask_padding_with_zero else 1] * padding_length)
            token_type_ids2 = token_type_ids2 + ([pad_token_segment_id] * padding_length)

        assert len(input_ids2) == max_length, "Error with input length {} vs {}".format(len(input_ids2), max_length)
        assert len(attention_mask2) == max_length, "Error with input length {} vs {}".format(len(attention_mask2),
                                                                                             max_length)
        assert len(token_type_ids2) == max_length, "Error with input length {} vs {}".format(len(token_type_ids2),
                                                                                             max_length)

        if output_mode == "classification":
            label = label_map[example.label]
        elif output_mode == "regression":
            label = float(example.label)
        else:
            raise KeyError(output_mode)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            logger.info("input_ids2: %s" % " ".join([str(x) for x in input_ids2]))
            logger.info("attention_mask2: %s" % " ".join([str(x) for x in attention_mask2]))
            logger.info("token_type_ids2: %s" % " ".join([str(x) for x in token_type_ids2]))
            logger.info("label: %s (id = %d)" % (example.label, label))

        features.append(
            InputFeatures(input_ids=input_ids,
                          attention_mask=attention_mask,
                          token_type_ids=token_type_ids,
                          input_ids2=input_ids2,
                          attention_mask2=attention_mask2,
                          token_type_ids2=token_type_ids2,
                          label=label))
    return features


output_modes = glue_output_modes
output_modes['snli'] = "classification"
output_modes['mini-snli'] = "classification"
output_modes['mini-snli-img'] = "classification"
output_modes['mini-qqp'] = glue_output_modes['qqp']
output_modes['mini-qnli'] = glue_output_modes['qnli']
output_modes['mini-mnli'] = glue_output_modes['mnli']

processors = glue_processors
processors['snli'] = SNLIProcessor
processors['mini-snli'] = SNLIProcessor
processors['mini-qqp'] = glue_processors['qqp']
processors['mini-qnli'] = glue_processors['qnli']
processors['mini-mnli'] = glue_processors['mnli']
processors['sts-b'] = StsbProcessor
