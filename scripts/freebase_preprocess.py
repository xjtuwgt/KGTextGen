import logging
from tqdm import tqdm
from os.path import join
import os
from evens import HOME_DATA_FOLDER, KG_DATA_FOLDER, PROCESSED_FOLDER
import argparse
from wikigraph.datautils import io_utils
from wikigraph.datautils import wikitext

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def default_pair_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--text_data_dir', type=str, default=join(HOME_DATA_FOLDER, 'wikitext-103'),
                        help='Path to the directory that contains the unzipped wikitext-103 data.')
    parser.add_argument('--kg_data_dir', type=str, default=KG_DATA_FOLDER,
                        help='Path to the directory that contains free-graph with different sizes.')
    parser.add_argument('--paired_file_path', type=str, default=join(PROCESSED_FOLDER, 'text_graph_pair'),
                        help='Path to the output (text, graph) pairs')
    parser.add_argument('--version', type=str, default='max256', choices=['max256', 'max512', 'max1024'],
                        help='Which version of paired data to use.')
    parser.add_argument('--subset', type=str, default='train', choices=['train', 'valid', 'test'],
                        help='train, valid, test')
    return parser


def pair_graphs_with_wikitext(subset: str, graph_dir: str, text_dir: str, output_dir: str):
    """Pair graphs with wikitext articles, and write to output directory."""
    logging.info('Pairing graphs from the %s set from %s with wikitext.',
                 subset, graph_dir)
    graphs = list(io_utils.graphs_from_file(join(graph_dir, f'{subset}.gz')))
    title2graph = {io_utils.normalize_freebase_string(g.title).replace(' ', ''): g for g in tqdm(graphs)}
    n_graphs = len(graphs)
    # Use raw version of the wikitext data as the tokenized version has <unk> in
    # titles which is bad for matching.  We will handle the <unk>s through the
    # tokenizer to make sure our data are equivalent to that of the tokenized
    # version of wikitext-103.
    wikitext_articles = list(wikitext.RawDataset(subset=subset, data_dir=text_dir))
    n_wiki = len(wikitext_articles)
    logging.info('Loaded %d graphs and %d wikitext articles in total.',  n_graphs, n_wiki)

    # Keep track of the article titles in the dataset.  Unfortunately wikitext-103
    # has about 1% of duplicated articles, we want to take care of that.
    retrieved_titles = set()
    pairs = []
    n_duplicates = 0
    for a in tqdm(wikitext_articles):
        title = wikitext.normalize_title(a.title).replace(' ', '')
        g = title2graph.get(title, None)
        if g is not None:
            if title not in retrieved_titles:
                retrieved_titles.add(title)
                pairs.append(io_utils.GraphTextPair(
                    text=a.text,
                    center_node=g.center,
                    title=g.title,
                    edges=g.edges))
            else:
                n_duplicates += 1

    n_pairs = len(pairs)
    logging.info('Matched %d/%d = %.1f%% of wikitext articles,'
                 ' and %d/%d = %.1f%% of graphs.',
                 n_pairs, n_wiki, float(n_pairs) / n_wiki * 100,
                 n_pairs, n_graphs, float(n_pairs) / n_graphs * 100)
    logging.info('Detected %d/%d = %.1f%% of duplicated wikitext articles.',
                 n_duplicates, n_wiki, float(n_duplicates) / n_wiki * 100)
    io_utils.write_pairs_to_gzip_txt_file(join(output_dir, f'{subset}.gz'), pairs)


def pair_graph_text_api(args):
    for key, value in vars(args).items():
        logging.info('Parameter {} = {}'.format(key, value))
    graph_dir = join(args.kg_data_dir, args.version)
    os.makedirs(args.paired_file_path, exist_ok=True)
    output_dir = join(args.paired_file_path, args.version)
    os.makedirs(output_dir, exist_ok=True)
    subset = args.subset
    pair_graphs_with_wikitext(subset=subset, graph_dir=graph_dir, text_dir=args.text_data_dir, output_dir=output_dir)
    return


if __name__ == '__main__':
    args = default_pair_arg_parser().parse_args()
    pair_graph_text_api(args=args)
