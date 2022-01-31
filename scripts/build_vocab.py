import collections
import enum
import os
from os.path import join
from typing import List, Tuple
import argparse
import logging
import pandas as pd
from tqdm import tqdm

from wikigraph.datautils import paired_dataset
from wikigraph.datautils import tokenizers
from wikigraph.datautils import wikitext
from evens import HOME_DATA_FOLDER, PROCESSED_FOLDER, KG_DATA_FOLDER
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def default_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=join(HOME_DATA_FOLDER, 'wikitext-103'),
                        help='Path to the directory that contains the unzipped wikitext-103 data.')
    parser.add_argument('--kg_pair_data_dir', type=str, default=join(PROCESSED_FOLDER, 'text_graph_pair'),
                        help='Path to the directory that contains the unzipped wikitext-103 data.')
    parser.add_argument('--vocab_file_path', type=str, default=join(PROCESSED_FOLDER, 'vocab'),
                        help='Path to the output vocab file.')
    parser.add_argument('--data_type', type=DatasetType, default=DatasetType.graph,
                        help='Path to the output vocab file.')
    parser.add_argument('--threshold', type=int, default=1, help='Frequency threshold for a word to be '
                                                                 'included in the vocabulary.')
    parser.add_argument('--version', type=str, default='max256', help='Which version of paired data to use.')
    return parser


class DatasetType(enum.Enum):
    text = 1
    graph = 2
    wikitext = 3


def get_vocab(dataset: wikitext.RawDataset) -> List[Tuple[str, int]]:
    """Build vocabulary, return (word, count) tuples sorted by count."""
    vocab = collections.defaultdict(int)
    count = 0
    for pair in tqdm(dataset):
        # pair (title, text)
        for t in pair.text.split(' '):
            if t:
                vocab[t] += 1
        count = count + 1
    logging.info('Number of words = {} extracted form {} documents'.format(len(vocab), count))
    return sorted(vocab.items(), key=lambda t: -t[1])


def write_vocab(vocab: List[Tuple[str, int]], output_path: str, out_file_name: str):
    """Write a vocab list to a file."""
    os.makedirs(output_path, exist_ok=True)
    vocab_df = pd.DataFrame(vocab, columns=['word', 'count'])
    vocab_file_name = join(output_path, out_file_name)
    vocab_df.to_csv(vocab_file_name, index=False, encoding='utf-8', errors='surrogatepass')


def build_wikitext_vocab(args):
    """
    Build vocab from wiki-text data
    :param args:
    :return:
    """
    logger.info('Loading the dataset.')
    dataset = wikitext.RawDataset(subset='train', data_dir=args.data_dir)
    logger.info('Building the vocab.')
    vocab = get_vocab(dataset)
    logging.info('Finished, vocab size %d, total number of tokens %d',
                 len(vocab), sum([c for _, c in vocab]))
    logging.info('Writing the vocab to %s', join(args.vocab_file_path, 'wikitext.vocab.csv'))
    write_vocab(vocab, args.vocab_file_path, 'wikitext.vocab.csv')


def build_graph_vocab(args):
    """Build vocabulary for graph data."""
    logger.info('Loading the dataset.')
    dataset = paired_dataset.ParsedDataset(
        subset='train', data_dir=args.kg_pair_data_dir, version=args.version)
    logger.info('Building graph vocab.')

    vocab = collections.defaultdict(int)
    max_graph_size = 0
    for pair in tqdm(dataset):
        graph = pair.graph
        if max_graph_size < len(graph.nodes()):
            max_graph_size = len(graph.nodes())
        for n in graph.nodes():
            for t in tokenizers.GraphTokenizer.split_node(n):
                if t:
                    vocab[t] += 1
        for _, _, e in graph.edges():
            for t in tokenizers.GraphTokenizer.split_edge(e):
                if t:
                    vocab[t] += 1
    logging.info('Max graph size = {}'.format(max_graph_size))
    vocab = sorted(vocab.items(), key=lambda t: -t[1])
    # vocab = [k for k, v in vocab if v >= args.threshold]
    vocab = [(k, v) for k, v in vocab if v >= args.threshold]
    logger.info('Finished, vocab size %d by filtered with %d', len(vocab), args.threshold)
    logger.info('Writing the vocab to %s.', join(args.vocab_file_path, 'graph.vocab.csv'))
    write_vocab(vocab, args.vocab_file_path, 'graph.vocab.csv')


def build_text_vocab(args):
    """Build vocabulary for the text part of the graph-to-text data."""
    logger.info('Loading the dataset.')
    dataset = paired_dataset.ParsedDataset(
        subset='train', data_dir=args.kg_pair_data_dir, version=args.version)
    logger.info('Building text vocab.')
    vocab = collections.defaultdict(int)
    count = 0
    for pair in tqdm(dataset):
        ## pair: (center_node, title, text, graph)
        for t in pair.text.split(' '):
            if t:
                vocab[t] += 1
        count = count + 1
    logging.info('Number of words = {} extracted form {} documents'.format(len(vocab), count))
    vocab = sorted(vocab.items(), key=lambda t: -t[1])
    logger.info('Finished, vocab size %d, total number of tokens %d.', len(vocab), sum([v for _, v in vocab]))
    vocab = [(k, v) for k, v in vocab if v >= args.threshold]
    logger.info('After filtering, vocab size %d.', len(vocab))
    logger.info('Writing the vocab to %s.', join(args.vocab_file_path, 'graph.text.vocab.csv'))
    write_vocab(vocab, args.vocab_file_path, 'graph.text.vocab.csv')


def main(args):
    for key, value in vars(args).items():
        logging.info('Parameter {} = {}'.format(key, value))
    if args.data_type == DatasetType.wikitext:
        build_wikitext_vocab(args)  ##  original wiki-text
    elif args.data_type == DatasetType.text:
        build_text_vocab(args)  ##  pair of (wiki-text, graph), wiki-text
    elif args.data_type == DatasetType.graph:
        build_graph_vocab(args)
    else:
        raise ValueError(f'Unknown data type {args.data_type}.')


def wikitext2graph(args):
    # wikidata = wikitext.RawDataset(subset='train', data_dir=args.data_dir)
    # wiki_title_dict = {}
    # for pair in tqdm(wikidata):
    #     wiki_title_dict[pair.title] = pair.text
    dataset = paired_dataset.ParsedDataset(
        subset='train', data_dir=args.kg_data_dir, version=args.version)
    for pair in tqdm(dataset):
        print(pair.title)
        print(pair.center_node)
        print(pair.text)
        print(pair.graph)
        # if title in wiki_title_dict:
        #     print(wiki_title_dict[title])
        #     print('+' * 10)
        #     print(text)
        #     print('*' * 10)


if __name__ == '__main__':
    args = default_arg_parser().parse_args()
    args.data_type = DatasetType.graph
    main(args=args)
    # wikitext2graph(args=args)