from evens import HOME_DATA_FOLDER
from os.path import join
import bert_score
import argparse
import torch

bert_score_example = join(HOME_DATA_FOLDER, 'bert_score_example')


def bert_score_func():
    torch.multiprocessing.set_sharing_strategy("file_system")

    parser = argparse.ArgumentParser("Calculate BERTScore")
    parser.add_argument(
        "--lang",
        type=str,
        default='en',
        help='two-letter abbreviation of the language (e.g., en) or "en-sci" for scientific text',
    )
    parser.add_argument(
        "-m", "--model", default=None, help="BERT model name (default: bert-base-uncased) or path to a pretrain model",
    )
    parser.add_argument("-l", "--num_layers", type=int, default=None, help="use first N layer in BERT (default: 8)")
    parser.add_argument("-b", "--batch_size", type=int, default=64, help="batch size (default: 64)")
    parser.add_argument("--nthreads", type=int, default=4, help="number of cpu workers (default: 4)")
    parser.add_argument("--idf", action="store_true", help="BERT Score with IDF scaling")
    parser.add_argument(
        "--rescale_with_baseline", action="store_true", help="Rescaling the numerical score with precomputed baselines",
    )
    parser.add_argument("--baseline_path", default=None, type=str, help="path of custom baseline csv file")
    parser.add_argument("--use_fast_tokenizer", action="store_false", help="whether to use HF fast tokenizer")
    parser.add_argument("-s", "--seg_level", action="store_true", help="show individual score of each pair")
    parser.add_argument("-v", "--verbose", action="store_true", help="increase output verbosity")
    parser.add_argument("-r", "--ref", type=str, default=join(bert_score_example, 'refs.txt'), help="reference file path(s) or a string")
    parser.add_argument(
        "-c", "--cand", type=str, default=join(bert_score_example, 'hyps.txt'), help="candidate (system outputs) file path or a string",
    )

    args = parser.parse_args()
    with open(args.cand) as f:
        cands = [line.strip() for line in f]
    refs = []
    with open(args.ref) as f:
        curr_ref = [line.strip() for line in f]
        assert len(curr_ref) == len(cands), f"# of sentences in {args.ref} doesn't match the # of candidates"
        refs.append(curr_ref)
    refs = list(zip(*refs))

    all_preds, hash_code = bert_score.score(
        cands,
        refs,
        model_type=args.model,
        num_layers=args.num_layers,
        verbose=args.verbose,
        idf=args.idf,
        batch_size=args.batch_size,
        lang=args.lang,
        return_hash=True,
        rescale_with_baseline=args.rescale_with_baseline,
        baseline_path=args.baseline_path,
        use_fast_tokenizer=args.use_fast_tokenizer,
    )
    avg_scores = [s.mean(dim=0) for s in all_preds]
    P = avg_scores[0].cpu().item()
    R = avg_scores[1].cpu().item()
    F1 = avg_scores[2].cpu().item()
    msg = hash_code + f" P: {P:.6f} R: {R:.6f} F1: {F1:.6f}"
    print(msg)
    if args.seg_level:
        ps, rs, fs = all_preds
        for p, r, f in zip(ps, rs, fs):
            print("{:.6f}\t{:.6f}\t{:.6f}".format(p, r, f))


def bert_score_visualization():
    torch.multiprocessing.set_sharing_strategy("file_system")

    parser = argparse.ArgumentParser("Visualize BERTScore")
    parser.add_argument("--lang", type=str, default="en", help="two-letter abbreviation of the language (e.g., en)")
    parser.add_argument("-m", "--model", default=None, help="BERT model name (default: bert-base-uncased)")
    parser.add_argument("-l", "--num_layers", type=int, default=None, help="use first N layer in BERT (default: 8)")
    parser.add_argument("-v", "--verbose", action="store_true", help="increase output verbosity")
    parser.add_argument("-r", "--ref", type=str, default="There are two bananas on the table.", help="reference sentence")
    parser.add_argument("-c", "--cand", type=str, default="On the table are two apples.", help="candidate sentence")
    parser.add_argument(
        "-f", "--file", type=str, default="visualize.png", help="name of file to save output matrix in",
    )
    parser.add_argument(
        "--rescale_with_baseline", action="store_true", help="Rescaling the numerical score with precomputed baselines",
    )
    parser.add_argument("--baseline_path", default=None, type=str, help="path of custom baseline csv file")

    args = parser.parse_args()

    bert_score.plot_example(
        args.cand,
        args.ref,
        model_type=args.model,
        lang=args.lang,
        num_layers=args.num_layers,
        fname=args.file,
        rescale_with_baseline=args.rescale_with_baseline,
        baseline_path=args.baseline_path,
    )