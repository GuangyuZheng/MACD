# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE (Bert, XLM, XLNet, RoBERTa)."""

from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import random
import json
import h5py

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, SubsetRandomSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

from tqdm import tqdm, trange

from transformers import (WEIGHTS_NAME, BertConfig, BertTokenizer)
from transformers import BertForSequenceClassification

from transformers import AdamW, get_linear_schedule_with_warmup

from utils.finetune_config import cfg, cfg_from_file
from utils.commons import MyEncoder
from utils.data import save_h5, HDF5Dataset

# from transformers import glue_compute_metrics as compute_metrics
# from transformers import glue_output_modes as output_modes
# from transformers import glue_processors as processors
from utils.finetune_data_bunch import compute_metrics, output_modes, processors
from transformers import glue_convert_examples_to_features as convert_examples_to_features

logger = logging.getLogger(__name__)

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig,)), ())

MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter(os.path.join(cfg.OUTPUT_DIR, 'logs'))

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    if args.num_samples < 0:
        train_sampler = RandomSampler(train_dataset)
    else:
        indices = list(range(args.num_samples))
        train_sampler = SubsetRandomSampler(indices)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
        args.logging_steps = len(train_dataloader) // args.gradient_accumulation_steps
        args.save_steps = args.logging_steps

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(t_total * 0.0),
                                                num_training_steps=t_total)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    best_performance_on_dev = -1
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        epoch_loss, epoch_step = 0.0, 0
        correct_num, total_num = 0, 0
        for step, batch in enumerate(epoch_iterator):
            model.train()
            if epoch_step != 0:
                epoch_iterator.set_description("Train Loss: %f, Acc: %f" % (epoch_loss / epoch_step,
                                                                            correct_num / total_num))
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'labels': batch[3]}
            inputs['token_type_ids'] = batch[2]
            # if cfg.TEXT.MODEL_TYPE != 'distilbert':
            #     inputs['token_type_ids'] = batch[2] if cfg.TEXT.MODEL_TYPE in ['bert',
            #                                                                    'xlnet'] else None  # XLM, DistilBERT and RoBERTa don't use segment_ids
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)
            logits = outputs[1].detach().cpu().numpy()
            preds = np.argmax(logits, axis=1)
            out_label_ids = inputs['labels'].detach().cpu().numpy()
            correct_num += (preds == out_label_ids).sum()
            total_num += len(preds)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            epoch_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1
                epoch_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logs = {}
                    if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer)
                        for key, value in results.items():
                            eval_key = 'eval_{}'.format(key)
                            logs[eval_key] = value

                        if 'acc_and_f1' in results:
                            tmp_performance = results['acc_and_f1']
                        elif 'corr' in results:
                            tmp_performance = results['corr']
                        elif 'acc' in results:
                            tmp_performance = results['acc']
                        elif 'mcc' in results:
                            tmp_performance = results['mcc']

                        if tmp_performance > best_performance_on_dev:
                            best_performance_on_dev = tmp_performance
                            # Save best model
                            output_dir = cfg.OUTPUT_DIR
                            if not os.path.exists(output_dir):
                                os.makedirs(output_dir)
                            model_to_save = model.module if hasattr(model,
                                                                    'module') else model  # Take care of distributed/parallel training
                            model_to_save.save_pretrained(output_dir)
                            tokenizer.save_pretrained(output_dir)
                            torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                            logger.info("Saving best model to %s", output_dir)

                    loss_scalar = (tr_loss - logging_loss) / args.logging_steps
                    learning_rate_scalar = scheduler.get_lr()[0]
                    logs['learning_rate'] = learning_rate_scalar
                    logs['loss'] = loss_scalar
                    logging_loss = tr_loss

                    for key, value in logs.items():
                        tb_writer.add_scalar(key, value, global_step)
                    print(json.dumps({**logs, **{'step': global_step}}, cls=MyEncoder))

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(cfg.OUTPUT_DIR, 'checkpoint-{}'.format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model,
                                                            'module') else model  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                    logger.info("Saving model checkpoint to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, prefix="", set_type="dev"):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = ("mnli", "mnli-mm") if cfg.TASK_NAME in ("mnli", "mini-mnli") else (cfg.TASK_NAME,)
    eval_outputs_dirs = (cfg.OUTPUT_DIR, cfg.OUTPUT_DIR + '-MM') if cfg.TASK_NAME in ("mnli", "mini-mnli") else (
        cfg.OUTPUT_DIR,)

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, set_type=set_type)

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # multi-gpu eval
        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'labels': batch[3]}
                inputs['token_type_ids'] = batch[2]
                # if cfg.TEXT.MODEL_TYPE != 'distilbert':
                #     inputs['token_type_ids'] = batch[2] if cfg.TEXT.MODEL_TYPE in ['bert',
                #                                                                    'xlnet'] else None  # XLM, DistilBERT and RoBERTa don't use segment_ids
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        if args.output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif args.output_mode == "regression":
            preds = np.squeeze(preds)
        result = compute_metrics(eval_task, preds, out_label_ids)
        results.update(result)

        output_eval_file = os.path.join(eval_output_dir, prefix, set_type + "_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results {} *****".format(prefix))
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    return results


def load_and_cache_examples(args, task, tokenizer, set_type='train', save_interval=10000):
    if args.local_rank not in [-1, 0] and set_type == 'train':
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(cfg.DATA_DIR, 'cached_{}_{}_{}_{}'.format(set_type,
                                                                                  list(filter(None,
                                                                                              cfg.TEXT.MODEL_NAME.split(
                                                                                                  '/'))).pop(),
                                                                                  str(cfg.TEXT.MAX_LEN), str(task)))

    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        # # features = torch.load(cached_features_file)
        # with h5py.File(cached_features_file, 'r') as dfile:
        #     all_input_ids = torch.tensor(dfile['all_input_ids'], dtype=torch.long)
        #     all_attention_mask = torch.tensor(dfile['all_attention_mask'], dtype=torch.long)
        #     all_token_type_ids = torch.tensor(dfile['all_token_type_ids'], dtype=torch.long)
        #     all_labels = torch.tensor(dfile['all_labels'],
        #                               dtype=torch.long if output_mode == 'classification' else torch.float)
    else:
        logger.info("Creating features from dataset file at %s", cfg.DATA_DIR)
        if os.path.isfile(cached_features_file):
            logger.info("Deleting existed cached file %s", cached_features_file)
            os.remove(cached_features_file)

        label_list = processor.get_labels()
        if task in ['mnli', 'mnli-mm'] and cfg.TEXT.MODEL_TYPE in ['roberta']:
            # HACK(label indices are swapped in RoBERTa pretrained model)
            label_list[1], label_list[2] = label_list[2], label_list[1]
        if set_type == 'train':
            examples = processor.get_train_examples(cfg.DATA_DIR)
        elif set_type == 'dev':
            examples = processor.get_dev_examples(cfg.DATA_DIR)
        else:
            examples = processor.get_test_examples(cfg.DATA_DIR)
        features = convert_examples_to_features(examples,
                                                tokenizer,
                                                label_list=label_list,
                                                max_length=cfg.TEXT.MAX_LEN,
                                                output_mode=output_mode,
                                                pad_on_left=bool(cfg.TEXT.MODEL_TYPE in ['xlnet']),
                                                # pad on the left for xlnet
                                                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                                pad_token_segment_id=4 if cfg.TEXT.MODEL_TYPE in ['xlnet'] else 0,
                                                )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            # torch.save(features, cached_features_file)
            all_input_ids = []
            all_attention_mask = []
            all_token_type_ids = []
            all_labels = []
            for (f_idx, f) in enumerate(features):
                all_input_ids.append(f.input_ids)
                all_attention_mask.append(f.attention_mask)
                all_token_type_ids.append(f.token_type_ids)
                all_labels.append(f.label)
                if len(all_input_ids) == save_interval:
                    logger.info("Saving example %d" % (f_idx + 1))
                    save_h5(cached_features_file, np.array(all_input_ids, dtype=np.long), 'input_ids')
                    save_h5(cached_features_file, np.array(all_attention_mask, dtype=np.long), 'attention_mask')
                    save_h5(cached_features_file, np.array(all_token_type_ids, dtype=np.long), 'token_type_ids')
                    if output_mode == "classification":
                        save_h5(cached_features_file, np.array(all_labels, dtype=np.long), "labels")
                    elif output_mode == "regression":
                        save_h5(cached_features_file, np.array(all_labels, dtype=np.float), "labels")
                    all_input_ids = []
                    all_attention_mask = []
                    all_token_type_ids = []
                    all_labels = []

            if len(all_input_ids) != 0:
                save_h5(cached_features_file, np.array(all_input_ids, dtype=np.long), 'input_ids')
                save_h5(cached_features_file, np.array(all_attention_mask, dtype=np.long), 'attention_mask')
                save_h5(cached_features_file, np.array(all_token_type_ids, dtype=np.long), 'token_type_ids')
                if output_mode == "classification":
                    save_h5(cached_features_file, np.array(all_labels, dtype=np.long), "labels")
                elif output_mode == "regression":
                    save_h5(cached_features_file, np.array(all_labels, dtype=np.float), "labels")

    if args.local_rank == 0 and set_type == 'train':
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    dataset = HDF5Dataset(cached_features_file, ['input_ids', 'attention_mask', 'token_type_ids', 'labels'])
    return dataset


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--cfg", default=None, type=str, required=True)
    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument("--multi_gpu", default=False, action='store_true')

    ## Few shot parameters
    parser.add_argument("--num_samples", type=int, default=-1)

    ## Other parameters
    parser.add_argument("--evaluate_during_training", default=True, action='store_true',
                        help="Rul evaluation during training at each logging step.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    # parser.add_argument("--warmup_steps", default=0, type=int,
    #                     help="Linear warmup over warmup_steps.")

    # parser.add_argument('--logging_steps', type=int, default=434,
    #                     help="Log every X updates steps.")
    # parser.add_argument('--save_steps', type=int, default=434,
    #                     help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")
    args = parser.parse_args()

    if args.cfg is not None:
        cfg_from_file(args.cfg)
        cfg.GPU_ID = args.gpu
        args.per_gpu_train_batch_size = cfg.BATCH_SIZE
        args.per_gpu_eval_batch_size = cfg.BATCH_SIZE * 4
        args.gradient_accumulation_steps = cfg.GRAD_ACCUM
        args.learning_rate = cfg.LR
        args.num_train_epochs = cfg.EPOCH

    # Prepare Few Shot settings
    if args.num_samples > 0:
        cfg.OUTPUT_DIR = cfg.OUTPUT_DIR + '-' + str(args.num_samples)

    if os.path.exists(cfg.OUTPUT_DIR) and os.listdir(
            cfg.OUTPUT_DIR) and cfg.TRAIN and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                cfg.OUTPUT_DIR))

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        if args.no_cuda:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:" + str(cfg.GPU_ID))
        if args.multi_gpu is True:
            args.n_gpu = torch.cuda.device_count()
        else:
            args.n_gpu = 1
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)

    # Prepare GLUE task
    cfg.TASK_NAME = cfg.TASK_NAME.lower()
    if cfg.TASK_NAME not in processors:
        raise ValueError("Task not found: %s" % (cfg.TASK_NAME))
    processor = processors[cfg.TASK_NAME]()
    args.output_mode = output_modes[cfg.TASK_NAME]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    cfg.TEXT.MODEL_TYPE = cfg.TEXT.MODEL_TYPE.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[cfg.TEXT.MODEL_TYPE]

    config = config_class.from_pretrained(cfg.TEXT.MODEL_NAME,
                                          num_labels=num_labels,
                                          finetuning_task=cfg.TASK_NAME,
                                          cache_dir=cfg.TEXT.CACHE_DIR if cfg.TEXT.CACHE_DIR else None)
    tokenizer = tokenizer_class.from_pretrained(cfg.TEXT.MODEL_NAME,
                                                do_lower_case=cfg.TEXT.LOWER_CASE,
                                                cache_dir=cfg.TEXT.CACHE_DIR if cfg.TEXT.CACHE_DIR else None)
    model = model_class.from_pretrained(cfg.TEXT.MODEL_NAME,
                                        from_tf=bool('.ckpt' in cfg.TEXT.MODEL_NAME),
                                        config=config,
                                        cache_dir=cfg.TEXT.CACHE_DIR if cfg.TEXT.CACHE_DIR else None)

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if cfg.TRAIN:
        train_dataset = load_and_cache_examples(args, cfg.TASK_NAME, tokenizer, set_type='train')
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    # if cfg.TRAIN and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
    #     # Create output directory if needed
    #     if not os.path.exists(cfg.OUTPUT_DIR) and args.local_rank in [-1, 0]:
    #         os.makedirs(cfg.OUTPUT_DIR)
    #
    #     logger.info("Saving model checkpoint to %s", cfg.OUTPUT_DIR)
    #     # Save a trained model, configuration and tokenizer using `save_pretrained()`.
    #     # They can then be reloaded using `from_pretrained()`
    #     model_to_save = model.module if hasattr(model,
    #                                             'module') else model  # Take care of distributed/parallel training
    #     model_to_save.save_pretrained(cfg.OUTPUT_DIR)
    #     tokenizer.save_pretrained(cfg.OUTPUT_DIR)
    #
    #     # Good practice: save your training arguments together with the trained model
    #     torch.save(args, os.path.join(cfg.OUTPUT_DIR, 'training_args.bin'))
    #
    #     # Load a trained model and vocabulary that you have fine-tuned
    #     model = model_class.from_pretrained(cfg.OUTPUT_DIR)
    #     tokenizer = tokenizer_class.from_pretrained(cfg.OUTPUT_DIR)
    #     model.to(args.device)

    # Evaluation
    results = {}
    test_results = {}
    if cfg.EVAL and args.local_rank in [-1, 0]:
        tokenizer = tokenizer_class.from_pretrained(cfg.OUTPUT_DIR, do_lower_case=cfg.TEXT.LOWER_CASE)
        checkpoints = [cfg.OUTPUT_DIR]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(cfg.OUTPUT_DIR + '/**/' + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split('/')[-1] if checkpoint.find('checkpoint') != -1 else ""

            model = model_class.from_pretrained(checkpoint)

            model.to(args.device)
            result = evaluate(args, model, tokenizer, prefix=prefix, set_type="dev")
            result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
            results.update(result)

            test_result = evaluate(args, model, tokenizer, prefix=prefix, set_type="test")
            test_result = dict((k + '_{}'.format(global_step), v) for k, v in test_result.items())
            test_results.update(test_result)

    return results


if __name__ == "__main__":
    main()
