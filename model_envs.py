from transformers import TransfoXLConfig, TransfoXLTokenizer, TransfoXLLMHeadModel
from transformers import XLNetConfig, XLNetLMHeadModel, XLNetTokenizer
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer


############################################################
# Model Related Global Varialbes
############################################################
MODEL_CLASSES = {
    'transfo-xl': (TransfoXLConfig, TransfoXLLMHeadModel, TransfoXLTokenizer),
    'xlnet': (XLNetConfig, XLNetLMHeadModel, XLNetTokenizer),
    'gpt2': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer)
}