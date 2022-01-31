import argparse
import torch
from evens import HOME_DATA_FOLDER
from wikigraph.gpu_utils import get_single_free_gpu
from wikigraph.utils import boolean_string
from text_generation_codes.lambada_dataloader import MAX_SENT_NUM, MAX_WORD_NUM
from text_generation_codes.gpt2_generation import MODEL_CLASSES
import numpy as np
import os
import random


def seed_everything(seed: int) -> int:
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    return seed


def default_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", default='gpt2', type=str,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--train_stage", default='false', type=boolean_string)
    parser.add_argument("--model_name_or_path", default='gpt2', type=str,
                        help="Path to pre-trained model or shortcut name selected in the list")
    parser.add_argument("--prompt", type=str, default="She loves every minute of it but now she's learning "
                                                      "how to wake up, open her")
    parser.add_argument("--padding_text", type=str, default="")
    parser.add_argument("--length", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="temperature of 0 implies greedy sampling")
    parser.add_argument("--repetition_penalty", type=float, default=1.0,
                        help="primarily useful for CTRL model; in that case, use 1.2")
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--train_epochs', type=int, default=100,
                        help="train_batch_size")
    parser.add_argument('--train_batch_size', type=int, default=8,
                        help="train_batch_size")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="train_batch_size")
    parser.add_argument('--sample_para_size', type=int, default=5,
                        help="sample para graphs from books")
    parser.add_argument('--fix_sample', type=boolean_string, default='true',
                        help="fix sample idx")
    parser.add_argument('--lr', type=float, default=2e-5,
                        help="learning rate")
    parser.add_argument('--weight_decay', type=float, default=1e-8,
                        help="learning rate")
    parser.add_argument('--warmup_steps', type=int, default=1000,
                        help="warm up steps")
    parser.add_argument('--max_ctx_word_num', type=int, default=MAX_WORD_NUM,
                        help="max ctx sequence length")
    parser.add_argument('--max_ctx_sent_num', type=int, default=MAX_SENT_NUM,
                        help="max ctx sequence length")
    parser.add_argument('--train_log_step', type=int, default=100,
                        help="valid log step")
    parser.add_argument('--valid_log_step', type=int, default=200,
                        help="valid log step")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--stop_word', type=boolean_string, default='true',
                        help="random seed for initialization")
    parser.add_argument('--local_rank', type=int, default=-1,
                        help="random seed for initialization")
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help="grad norm")
    parser.add_argument('--stanford', type=boolean_string, default='false',
                        help="random seed for initialization")
    parser.add_argument('--data_parallel', type=boolean_string, default='false',
                        help="random seed for initialization")
    parser.add_argument('--stop_token', type=str, default=None,
                        help="Token at which text generation is stopped")
    parser.add_argument('--data_type', type=str, default='validation', choices=['train', 'validation', 'test'],
                        help="Token at which text generation is stopped")
    ##++++++++++++++++++
    parser.add_argument("--word_drop_ratio", type=float, default=0.0)
    parser.add_argument("--drop_prob", type=float, default=0.0)
    parser.add_argument("--beta_drop", type=boolean_string, default='true')
    ##++++++++++++++++++
    args = parser.parse_args()
    return args


def completed_argument_parser(args):
    seed_everything(seed=args.seed)
    if HOME_DATA_FOLDER.startswith('/dfs/scratch0'):
        args.stanford = 'true'
    if args.local_rank == -1:
        if args.stanford:
            if torch.cuda.is_available():
                gpu_idx, _ = get_single_free_gpu()
                device = torch.device("cuda:{}".format(gpu_idx) if torch.cuda.is_available() else "cpu")
            else:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.data_parallel:
            args.n_gpu = torch.cuda.device_count()
        else:
            args.n_gpu = 1
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device
    return args