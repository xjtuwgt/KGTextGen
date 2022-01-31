from scripts.freebase_preprocess import default_pair_arg_parser
from wikigraph.datautils.io_utils import read_pairs_from_gzip_txt_file
from wikigraph.datautils.paired_dataset import Graph
from wikigraph.datautils.io_utils import normalize_freebase_string
from os.path import join


def symmetry_equal_char_count(text_line):
    counter = 0
    n = len(text_line)
    index = 0
    while index < n/2 and (text_line[index] == text_line[n - index - 1]) and (text_line[index] in {' ', '='}):
        counter = counter + 1
        index = index + 1
    return counter


def wiki_text_paragraph_parser(text: str):
    text_lines = text.split('\n')
    line_idx = 0
    paragraph_list = []
    while line_idx < len(text_lines):
        text_line = text_lines[line_idx].strip()
        if not text_line:
            line_idx = line_idx + 1
            continue
        else:
            if text_line.startswith('= ') and text_line.endswith(' ='):
                maker_count = symmetry_equal_char_count(text_line=text_line)
                assert maker_count % 2 == 0, 'marker count = {} \n text = {}'.format(maker_count, text_line)
                text_line_len = len(text_line)
                para_title = text_line[maker_count:(text_line_len - maker_count)]
                para_title_level = maker_count // 2
                para_index = line_idx + 1
                para_text = []
                while para_index < len(text_lines):
                    para_text_line = text_lines[para_index].strip()
                    if para_text_line.startswith('= ') and para_text_line.endswith(' ='):
                        break
                    else:
                        if para_text_line:
                            para_text.append(para_text_line)
                        para_index = para_index + 1
                line_idx = para_index
                paragraph_list.append((para_title_level - 1, para_title, para_text))
            else:
                print('error')
    return paragraph_list


def wiki_text_ner_extractor(text_line: str, graph: Graph):
    def node_filter(node: str):
        return node.startswith('ns/m') or node.startswith('"/') or len(node.split()) >= 10
    node_num = len(graph.nodes())

    def node_normalization(node: str):
        return normalize_freebase_string(node)

    nodes = [node_normalization(_) for _ in graph.nodes() if not node_filter(node=_)]
    red_node_num = len(nodes)
    nodes = list(set(nodes))
    print(node_num, red_node_num, len(nodes))

    assert len(nodes) >= 1
    print(nodes)
    paragraph_list = wiki_text_paragraph_parser(text=text_line)
    true_count = 0
    for node in nodes:
        node_true_count = 0
        for para in paragraph_list:
            for text in para[-1]:
                if node in text:
                    node_true_count = node_true_count + 1
        if node_true_count > 0:
            true_count = true_count + 1



    # print(len(nodes), true_count)


    return true_count


args = default_pair_arg_parser().parse_args()
args.subset = 'valid'

for key, value in vars(args).items():
    print('Hyper-parameter {}: {}'.format(key, value))

data = read_pairs_from_gzip_txt_file(join(args.paired_file_path, args.version, args.subset + '.gz'))

count = 0
for x in data:
    count = count + 1
    # print(x)
    graph_i = Graph.from_edges(x.edges)
    true_c = wiki_text_ner_extractor(text_line=x.text, graph=graph_i)
    # print(true_c)
    # y = wiki_text_paragraph_parser(text=x.text)
    # print(x.title)
    # for a, b, c in y:
    #     print(a, b)
    # print('*' * 10)

    # if count == 2:
    #     ns_node_count = 0
    #     print(len(graph.nodes()))
    #     for node in graph.nodes():
    #         print(node)
    #         if node.startswith('ns'):
    #             ns_node_count = ns_node_count + 1
    #     # print(graph.edges())
    #     print(ns_node_count)
    #
    # # for edge in graph.edges():
    # #     print(edge[2])
    #


# print(len(list(data)))

