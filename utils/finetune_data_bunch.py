import os
from transformers import glue_compute_metrics
from transformers import glue_output_modes
from transformers import glue_processors
from transformers.data.processors.glue import StsbProcessor as STSBP
from transformers.data import DataProcessor, InputExample
from transformers.data.metrics import simple_accuracy
from utils.pretrain_data_utils import InputExample as ImgInputExample


def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == "snli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == 'mini-snli':
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == 'mini-snli-img':
        return {"acc": simple_accuracy(preds, labels)}
    else:
        if 'mini' in task_name:
            task_name = task_name.split('mini-')[-1]
        elif 'kd-filtered' in task_name:
            task_name = task_name.split('kd-filtered')[-1]
        elif 'filtered' in task_name:
            task_name = task_name.split('filtered-')[-1]
        return glue_compute_metrics(task_name, preds, labels)


class SNLIProcessor(DataProcessor):
    """Processor for the RTE data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(tensor_dict['idx'].numpy(),
                            tensor_dict['sentence1'].numpy().decode('utf-8'),
                            tensor_dict['sentence2'].numpy().decode('utf-8'),
                            str(tensor_dict['label'].numpy()))

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "snli_1.0_train.txt")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "snli_1.0_dev.txt")), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "snli_1.0_test.txt")), "test")

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[5]
            text_b = line[6]
            label = line[0]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class StsbProcessor(STSBP):
    """Processor for the STS-B data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(tensor_dict['idx'].numpy(),
                            tensor_dict['sentence1'].numpy().decode('utf-8'),
                            tensor_dict['sentence2'].numpy().decode('utf-8'),
                            str(tensor_dict['label'].numpy()))

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "original", "sts-train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "original", "sts-dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "original", "sts-test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return [None]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            # if i == 0:
            #     continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[5]
            text_b = line[6]
            label = line[4]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class ImageSNLIProcessor(DataProcessor):

    def get_train_examples(self, data_dir, image_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "snli_1.0_train.txt")), "train", image_dir)

    def get_dev_examples(self, data_dir, image_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "snli_1.0_dev.txt")), "dev", image_dir)

    def get_labels(self):
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, lines, set_type, image_dir):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[5]
            text_b = line[6]
            image = line[7].strip().split('#')[0]
            image = os.path.join(image_dir, image)
            label = line[0]
            examples.append(
                ImgInputExample(guid=guid, text_a=text_a, text_b=text_b, image=image, label=label))
        return examples


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
processors['mini-snli-img'] = ImageSNLIProcessor
processors['mini-qqp'] = glue_processors['qqp']
processors['mini-qnli'] = glue_processors['qnli']
processors['mini-mnli'] = glue_processors['mnli']
processors['sts-b'] = StsbProcessor
