from wikigraph.argument_parser import generation_default_parser
from wikigraph.utils import seed_everything
from model_envs import MODEL_CLASSES
import torch

PREFIX = """This book is good."""


def prepare_transfoxl_input(args, prompt_text):
    prefix = args.prefix if args.prefix else args.padding_text if args.padding_text else PREFIX
    prompt_text = prefix + prompt_text
    return prompt_text

args = generation_default_parser().parse_args()
args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
seed_everything(seed=args.seed)

for key, value in vars(args).items():
    print(key, value)

# Initialize the model and tokenizer
try:
    args.model_type = args.model_type.lower()
    _, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
except KeyError:
    raise KeyError("the model {} you specified is not supported. You are welcome to add it and open a PR :)")

tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
model = model_class.from_pretrained(args.model_name_or_path)
model.to(args.device)

# prompt_text = args.prompt if args.prompt else input("Model prompt >>> ")
# print(prompt_text)

# preprocessed_prompt_text = prepare_transfoxl_input(args, '')
# print(preprocessed_prompt_text)

prompt_text = PREFIX

encoded_prompt = tokenizer.encode(prompt_text, add_special_tokens=False, return_tensors="pt")
# print(encoded_prompt)
encoded_prompt = encoded_prompt.to(args.device)
if encoded_prompt.size()[-1] == 0:
    input_ids = None
else:
    input_ids = encoded_prompt
print(input_ids)

output_sequences = model.generate(
        input_ids=input_ids,
        max_length=args.length + len(encoded_prompt[0]),
        temperature=args.temperature,
        top_k=args.k,
        top_p=args.p,
        repetition_penalty=args.repetition_penalty,
        do_sample=True,
        num_return_sequences=args.num_return_sequences,
    )

print(output_sequences)

generated_sequences = []

for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
    print(f"=== GENERATED SEQUENCE {generated_sequence_idx + 1} ===")
    generated_sequence = generated_sequence.tolist()

    # Decode text
    text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)

    # Remove all text after the stop token
    text = text[: text.find(args.stop_token) if args.stop_token else None]

    # Add the prompt at the beginning of the sequence. Remove the excess text that was used for pre-processing
    total_sequence = (
            prompt_text + text[len(tokenizer.decode(encoded_prompt[0], clean_up_tokenization_spaces=True)):]
    )

    generated_sequences.append(total_sequence)
    print(total_sequence)
