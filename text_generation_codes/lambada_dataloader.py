import itertools
import string

import numpy as np
import pandas as pd
import torch
from pandas import DataFrame
from datasets import load_dataset
from nltk.tokenize import sent_tokenize, word_tokenize
from tqdm import tqdm
from time import time
from evens import LAMBADA_DATASET_FOLDER
from numpy import random
from scipy.stats import beta
import truecase
import os
from text_generation_codes.gpt2_model_envs import stop_words
from nltk.tokenize.treebank import TreebankWordDetokenizer
from transformers import GPT2Tokenizer
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler

detokenizer = TreebankWordDetokenizer()

MAX_SENT_NUM, MIN_SENT_NUM = 12, 2
MAX_WORD_NUM, MIN_WORD_NUM = 203, 58
IGNORE_INDEX = -100


def detokenize_text(text: str):
    return truecase.get_true_case(detokenizer.detokenize(text.split()).replace(' .', '.')).strip()


class LambadaTrainDataSet(Dataset):
    def __init__(self, dataset: DataFrame, tokenizer: GPT2Tokenizer, word_drop_ratio=0.25, drop_prob=0.0,
                 beta_drop=True, max_word_num=MAX_WORD_NUM, max_sent_num=MAX_SENT_NUM, min_word_num=58,
                 min_sent_num=2, detoken_text=True, max_len=384, fix_sample=False):
        self.examples = dataset
        self.detoken_text = detoken_text
        self.example_len = self.examples.shape[0]
        self.max_word_num = max_word_num
        self.min_word_num = min_word_num
        self.max_sent_num = max_sent_num
        self.min_sent_num = min_sent_num
        self.word_drop_ratio = word_drop_ratio
        self.drop_prob = drop_prob
        self.beta_drop = beta_drop
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.fix_sample = fix_sample

    def __len__(self):
        return self.example_len

    def __getitem__(self, idx):
        case = self.examples.iloc[idx].to_dict()
        word_list, end_word_idx = train_lambada_case_generation(case=case,
                                                                max_word_num=self.max_word_num,
                                                                max_sent_num=self.max_sent_num,
                                                                fix_sample=self.fix_sample,
                                                                detokenized=self.detoken_text)
        assert len(word_list) >= 2
        ctx_word_list = word_list[:end_word_idx]
        target_text = word_list[end_word_idx]
        # print(target_text, len(ctx_word_list))
        random_number = random.random()
        if self.word_drop_ratio > 0 and self.beta_drop:
            a = max(1.0, self.word_drop_ratio / (1 - self.word_drop_ratio))
            b = max(1.0, (1 - self.word_drop_ratio) / self.word_drop_ratio)
            sent_drop_prob = beta.rvs(a * self.word_drop_ratio, b * self.word_drop_ratio)
        else:
            sent_drop_prob = self.word_drop_ratio
        if sent_drop_prob > 0 and random_number > self.drop_prob:
            ctx_word_list = word_drop(word_list=ctx_word_list, drop_ratio=sent_drop_prob)
        ctx_text = ' '.join(ctx_word_list)
        ctx_text_ids = self.tokenizer.encode('<|endoftext|>' + ctx_text, truncation=True)
        target_text_ids = self.tokenizer.encode(' ' + target_text + '<|endoftext|>', truncation=True)
        pad_ids = self.tokenizer.encode('<|endoftext|>', truncation=True)
        ctx_len = len(ctx_text_ids)
        target_len = len(target_text_ids)
        assert target_len > 0
        input_ids = ctx_text_ids + target_text_ids
        attn_mask = [1] * len(input_ids)
        attn_mask = attn_mask[:self.max_len]
        input_ids = input_ids[:self.max_len]
        labels = [IGNORE_INDEX] * ctx_len + target_text_ids
        labels = labels[:self.max_len]
        input_len = len(input_ids)
        if input_len < self.max_len:
            pad_len = self.max_len - input_len
            input_ids = pad_ids * pad_len + input_ids
            labels = [IGNORE_INDEX] * pad_len + labels
            attn_mask = [0] * pad_len + attn_mask
        input_ids = torch.as_tensor(input_ids, dtype=torch.long)
        attn_mask = torch.as_tensor(attn_mask, dtype=torch.long)
        labels = torch.as_tensor(labels, dtype=torch.long)
        return {'input_ids': input_ids, 'attn_mask': attn_mask, 'labels': labels}


def train_lambada_case_generation(case: dict, max_word_num: int = MAX_WORD_NUM, max_sent_num: int = MAX_SENT_NUM,
                                  fix_sample=False, detokenized=True):
    ctx_word_array = case['ctx']
    ctx_sent_len_array = case['ctx_len']
    ctx_sent_num = case['sent_num']
    if fix_sample:
        sample_start_idxs = case['samp_start_idx']
        start_sent_idx = random.choice(sample_start_idxs, 1)[0]
    else:
        if ctx_sent_num <= max_sent_num:
            start_sent_idx = 0
        else:
            start_sent_idx = random.randint(0, ctx_sent_num - max_sent_num, 1)[0]
    ctx_word_num = random.randint(MIN_WORD_NUM + 10, max_word_num + 30, 1)[0]
    end_sent_idx = start_sent_idx
    word_num = ctx_sent_len_array[end_sent_idx]
    extracted_ctx_words = [ctx_word_array[end_sent_idx].tolist()]
    while end_sent_idx < ctx_sent_num - 1 and word_num <= ctx_word_num:
        end_sent_idx = end_sent_idx + 1
        word_num = word_num + ctx_sent_len_array[end_sent_idx]
        extracted_ctx_words.append(ctx_word_array[end_sent_idx].tolist())
    if len(extracted_ctx_words) < MIN_SENT_NUM:
        end_sent_idx = end_sent_idx + 1
        if end_sent_idx < ctx_sent_num:
            word_num = word_num + ctx_sent_len_array[end_sent_idx]
            extracted_ctx_words.append(ctx_word_array[end_sent_idx].tolist())
    ctx_word_list = list(itertools.chain(*extracted_ctx_words))
    assert len(ctx_word_list) == word_num and word_num >= 2
    if detokenized:
        ctx_text = ' '.join(ctx_word_list)
        ctx_text = detokenize_text(text=ctx_text)
        ctx_word_list = ctx_text.split()
    end_word_idx = len(ctx_word_list) - 1
    while end_word_idx >= 0 and ((ctx_word_list[end_word_idx].strip() in string.punctuation) or
                                 (ctx_word_list[end_word_idx].strip() in {'``'})):
        end_word_idx = end_word_idx - 1
    print(ctx_word_list[end_word_idx])
    return ctx_word_list, end_word_idx


def word_drop(word_list: list, drop_ratio: float):
    keep_word_list = []
    for word in word_list:
        rand_s_i = random.rand()
        if rand_s_i >= drop_ratio:
            keep_word_list.append(word)
    if len(keep_word_list) == 0:
        return word_list
    return keep_word_list


class LambadaDevTestDataSet(Dataset):
    def __init__(self, dataset, detoken_text=True, bos_token="<|endoftext|>"):
        self.examples = dataset
        self.example_len = len(self.examples)
        self.detoken_text = detoken_text
        self.bos_token = bos_token

    def __len__(self):
        return self.example_len

    def __getitem__(self, idx):
        case = self.examples[idx]
        text_sentences = sent_tokenize(text=case['text'])
        ctx_sents = text_sentences[:-1]
        last_sent_words = word_tokenize(text_sentences[-1])
        target_sent_words = last_sent_words[:-1]
        target_word = last_sent_words[-1]
        prompt_words = ctx_sents + target_sent_words
        prompt_text = ' '.join(prompt_words)
        if self.detoken_text:
            prompt_text = self.bos_token + detokenize_text(prompt_text)
        exam_case = {'id': idx, 'prompt_text': prompt_text, 'target_word': target_word}
        return exam_case

    @staticmethod
    def valid_collection_func(data):
        assert len(data[0]) == 3
        batch_idx = [_['id'] for _ in data]
        batch_prompt_text = [_['prompt_text'] for _ in data]
        batch_target_word = [_['target_word'] for _ in data]
        batch = {'id': batch_idx, 'prompt_text': batch_prompt_text, 'target_word': batch_target_word}
        return batch


def lambada_train_data_preprocess():
    dataset = load_dataset(path='lambada', cache_dir=LAMBADA_DATASET_FOLDER)
    train_examples = dataset['train']
    sent_nums = []
    token_sequences = []
    for idx, case in tqdm(enumerate(train_examples)):
        case_text = case['text']
        text_sentences = sent_tokenize(text=case_text)
        sent_num = len(text_sentences)
        sent_nums.append(sent_num)
        if sent_num > 0:
            text_toekens = [word_tokenize(text=sent) for sent in text_sentences]
            token_lens = [len(_) for _ in text_toekens]
            token_len_sum = sum(token_lens)
            case_token = (idx, text_toekens, token_lens, token_len_sum, sent_num)
            token_sequences.append(case_token)
    data_frame = pd.DataFrame(data=token_sequences, columns=['id', 'ctx', 'ctx_len', 'total_len', 'sent_num'])
    start_time = time()
    df_file_name = os.path.join(LAMBADA_DATASET_FOLDER, 'cached_lambada_train.feather')
    data_frame.to_feather(df_file_name)
    print('Saving {} to {} in {} seconds'.format(data_frame.shape, df_file_name, time() - start_time))


def load_lambada_tokenized_feather_data():
    start_time = time()
    df_file_name = os.path.join(LAMBADA_DATASET_FOLDER, 'cached_lambada_train.feather')
    data_frame = pd.read_feather(df_file_name, columns=None, use_threads=True)
    print('Loading {} records from {} takes {:.4f} seconds'.format(data_frame.shape, df_file_name, time() - start_time))
    return data_frame


def lambada_dev_test_data_preprocess():
    from transformers import GPT2Tokenizer
    dataset = load_dataset(path='lambada', cache_dir=LAMBADA_DATASET_FOLDER)
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    valid_examples = dataset['validation']
    test_examples = dataset['test']
    sent_num_list = []
    word_num_list = []
    gpt2_word_sum_list = []
    for example in tqdm(valid_examples):
        case_text = example['text']
        sents = sent_tokenize(text=case_text)
        sent_words = [word_tokenize(_) for _ in sents]
        gpt2_words = [gpt2_tokenizer.tokenize(_) for _ in sents]
        gpt2_word_nums = [len(_) for _ in gpt2_words]
        word_nums = [len(_) for _ in sent_words]
        word_sum = sum(word_nums)
        gpt2_word_sum = sum(gpt2_word_nums)
        sent_num_list.append(len(sents))
        word_num_list.append(word_sum)
        gpt2_word_sum_list.append(gpt2_word_sum)

    for example in tqdm(test_examples):
        case_text = example['text']
        sents = sent_tokenize(text=case_text)
        sent_words = [word_tokenize(_) for _ in sents]
        gpt2_words = [gpt2_tokenizer.tokenize(_) for _ in sents]
        gpt2_word_nums = [len(_) for _ in gpt2_words]
        word_nums = [len(_) for _ in sent_words]
        word_sum = sum(word_nums)
        gpt2_word_sum = sum(gpt2_word_nums)
        sent_num_list.append(len(sents))
        word_num_list.append(word_sum)
        gpt2_word_sum_list.append(gpt2_word_sum)

    sent_num_array = np.array(sent_num_list)
    word_num_array = np.array(word_num_list)
    gpt2_word_num_array = np.array(gpt2_word_sum_list)
    print('sent', sent_num_array.max(), sent_num_array.mean(), sent_num_array.min())
    print('word', word_num_array.max(), word_num_array.mean(), word_num_array.min())
    print('gpt2 word', gpt2_word_num_array.max(), gpt2_word_num_array.mean(), gpt2_word_num_array.min())


def sample_sent_idx(data_frame: DataFrame, k=20, max_sent_num=MAX_SENT_NUM):
    def row_sample(row):
        ctx_sent_num = row['sent_num']
        if ctx_sent_num <= max_sent_num:
            start_sent_idx_array = np.array([0] * k)
        else:
            start_sent_idx_array = random.randint(0, ctx_sent_num - max_sent_num, k)
        return start_sent_idx_array

    data_frame['samp_start_idx'] = data_frame.apply(lambda row: row_sample(row), axis=1)
    return data_frame


class LambadaHelper(object):
    def __init__(self, config):
        self.config = config
        self.dataset = load_dataset(path='lambada', cache_dir=LAMBADA_DATASET_FOLDER)
        print('data set key words:', self.dataset.keys())
        print('Train size = {}'.format(len(self.dataset['train'])))
        print('Validation size = {}'.format(len(self.dataset['validation'])))
        print('Test size = {}'.format(len(self.dataset['test'])))
        print('*' * 75)
        self.valid_dataset = LambadaDevTestDataSet(dataset=self.dataset['validation'])
        self.test_dataset = LambadaDevTestDataSet(dataset=self.dataset['test'])
        if self.config.train_stage:
            tokenizer = GPT2Tokenizer.from_pretrained(self.config.model_name_or_path, bos_token='<|endoftext|>',
                                                      eos_token='<|endoftext|>', pad_token='<|endoftext|>')

            train_data_frame = load_lambada_tokenized_feather_data()
            if self.config.sample_para_size > 0:
                train_data_frame = sample_sent_idx(data_frame=train_data_frame,
                                                   k=self.config.sample_para_size,
                                                   max_sent_num=MAX_SENT_NUM)
                print('Train data frame size = {}'.format(train_data_frame.shape))

            self.train_dataset = LambadaTrainDataSet(dataset=train_data_frame,
                                                     word_drop_ratio=self.config.word_drop_ratio,
                                                     drop_prob=self.config.drop_prob,
                                                     beta_drop=self.config.beta_drop,
                                                     max_word_num=self.config.max_ctx_word_num,
                                                     max_sent_num=self.config.max_ctx_sent_num,
                                                     fix_sample=self.config.fix_sample,
                                                     tokenizer=tokenizer)
        else:
            self.train_dataset = None

    def valid_data_loader(self):
        valid_loader = DataLoader(dataset=self.valid_dataset,
                                  drop_last=False,
                                  batch_size=1,
                                  collate_fn=LambadaDevTestDataSet.valid_collection_func)
        return valid_loader

    def test_data_loader(self):
        test_loader = DataLoader(dataset=self.test_dataset,
                                 drop_last=False,
                                 batch_size=1,
                                 collate_fn=LambadaDevTestDataSet.valid_collection_func)
        return test_loader

    def train_data_loader(self):
        train_loader = DataLoader(
            dataset=self.train_dataset,  # The training samples.
            # sampler=RandomSampler(self.train_dataset),  # Select batches randomly
            # batch_size=self.config.train_batch_size,  # Trains with this batch size.
            batch_size = 1,
            shuffle=True,
        )
        return train_loader


if __name__ == '__main__':
    from text_generation_codes.argument_lambada_parser import default_argument_parser, completed_argument_parser

    args = completed_argument_parser(default_argument_parser())
    args.train_stage = 'true'
    datahelper = LambadaHelper(config=args)
    train_loader = datahelper.train_data_loader()
    for batch_idx, batch in enumerate(train_loader):
        print(batch['input_ids'].shape)
        # print(batch['labels'].shape)
        # print(batch['labels'])
        # print(batch_idx)

    # lambada_dev_test_data_preprocess()

    # case_dict = datahelper.train_dataset.examples.iloc[0].to_dict()
    #
    # train_lambada_case_generation(case=case_dict)
