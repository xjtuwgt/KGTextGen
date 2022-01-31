import argparse
from model_envs import MODEL_CLASSES


def generation_default_parser():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--model_type", default='transfo-xl', type=str,
    #                     help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    # parser.add_argument("--model_name_or_path", default='transfo-xl-wt103', type=str,
    #                     help="Path to pre-trained model or shortcut name selected "
    #                          "in the list: " + ", ".join(MODEL_CLASSES.keys()))

    parser.add_argument("--model_type", default='gpt2', type=str,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default='gpt2', type=str,
                        help="Path to pre-trained model or shortcut name selected "
                             "in the list: " + ", ".join(MODEL_CLASSES.keys()))

    parser.add_argument("--prompt", type=str, default="I have a dream.")
    parser.add_argument("--length", type=int, default=20)
    parser.add_argument("--stop_token", type=str, default=None, help="Token at which text generation is stopped")

    parser.add_argument("--temperature", type=float, default=1.0,
                        help="temperature of 1.0 has no effect, lower tend toward greedy sampling")
    parser.add_argument(
        "--repetition_penalty", type=float, default=1.0, help="primarily useful for CTRL model; in that case, use 1.2")
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--p", type=float, default=0.9)

    parser.add_argument("--prefix", type=str, default="", help="Text added prior to input.")
    parser.add_argument("--padding_text", type=str, default="", help="Deprecated, the use of `--prefix` is preferred.")
    parser.add_argument("--xlm_language", type=str, default="", help="Optional language when used with the XLM model.")

    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--num_return_sequences", type=int, default=5, help="The number of samples to generate.")
    return parser