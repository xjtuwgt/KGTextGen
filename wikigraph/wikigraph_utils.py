from evens import PROCESSED_FOLDER
from wikigraph.datautils import tokenizers
from os.path import join
import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)


VOCAB_FILES_MAP = {
    'wikitext': join(PROCESSED_FOLDER, 'vocab', 'wikitext.vocab.csv'),
    'freebase2wikitext': join(PROCESSED_FOLDER, 'vocab', 'graph.text.vocab.csv'),
    'graphtext': join(PROCESSED_FOLDER, 'vocab', 'graph.vocab.csv')
}


def init_tokenizer(dataset_name):
  """Initialie the tokenizer."""
  logging.info('Loading tokenizer...')
  tokenizer = tokenizers.WordTokenizer(VOCAB_FILES_MAP[dataset_name])
  logging.info('Vocab size: %d', tokenizer.vocab_size)
  return tokenizer


def init_graph_tokenizer(dataset_name):
    """Initialie the tokenizer."""
    logging.info('Loading tokenizer...')
    tokenizer = tokenizers.GraphTokenizer(VOCAB_FILES_MAP[dataset_name])
    logging.info('Vocab size: %d', tokenizer.vocab_size)
    return tokenizer


if __name__ == '__main__':
    init_tokenizer(dataset_name='wikitext')
    # init_graph_tokenizer(dataset_name='graphtext')
