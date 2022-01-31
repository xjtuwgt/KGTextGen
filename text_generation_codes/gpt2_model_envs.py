from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
from nltk.corpus import stopwords

stops = set(stopwords.words('english'))
stop_words = stops.union(['for', 'though', 'please', 'still', 'oh', 'going', 'let'])

# print(stops)
print(('for' in stops))
MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop
MODEL_CLASSES = {'gpt2': (GPT2LMHeadModel, GPT2Tokenizer, GPT2Config)}

