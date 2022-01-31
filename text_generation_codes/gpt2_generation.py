from tqdm import trange
from text_generation_codes.gpt2_model_envs import MODEL_CLASSES
from text_generation_codes.gpt2_model_envs import stop_words as stops
import torch
import torch.nn.functional as F
from torch import Tensor
import string


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        # print('single idx', indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits


def top_k_top_p_filtering_batch(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    assert logits.dim() == 2
    top_k = min(top_k, logits.size(-1))  # Safety check
    # print('batch', top_k, top_p, filter_value)
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
        sorted_indices_to_remove[:, 0] = 0
        for i in range(logits.shape[0]):
            indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
            logits[i][indices_to_remove] = filter_value
        # print(sorted_indices_to_remove.shape)
        # indices_to_remove_1 = sorted_indices[0][sorted_indices_to_remove[0]]
        #
        # print('batch idx 1', indices_to_remove_1)
        #
        # indices_to_remove = sorted_indices_to_remove.gather(1, sorted_indices)
        # print('batch idx', indices_to_remove[0].nonzero(as))
        # print('batch 2 shape', indices_to_remove.shape)
        # # indices_to_remove = sorted_indices[sorted_indices_to_remove]
        # logits[indices_to_remove] = filter_value
    return logits


def normalize_text(text: str) -> str:
    def remove_punc(x):
        return x.strip(string.punctuation)

    def lower(x):
        return x.lower()

    return lower(remove_punc(x=text)).strip()


class NextWordPrediction:
    def __init__(self, config, model=None):
        self.config = config
        self.model_type = self.config.model_type.lower()
        model_class, tokenizer_class, _ = MODEL_CLASSES[self.model_type]
        self.tokenizer = tokenizer_class.from_pretrained(self.config.model_name_or_path)
        if model is None:
            self.model = model_class.from_pretrained(self.config.model_name_or_path)
        else:
            self.model = model
            print('Tuned model')
        self.model.to(self.config.device)
        self.tokenizer.padding_side = "left"
        # Define PAD Token = EOS Token = 50256
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.model.config.eos_token_id

    def sample_sequence_single(self, length: int, context_input: Tensor, num_samples=1,
                               temperature=1, top_k=0, top_p=0.0, repetition_penalty=1.0):
        assert context_input.dim() == 2 and context_input.shape[0] == 1
        context = context_input.repeat(num_samples, 1)
        generated = context
        with torch.no_grad():
            for _ in trange(length):
                inputs = {'input_ids': generated}
                outputs = self.model(**inputs)
                # print(len(outputs), outputs[0].shape, type(outputs[1]), len(outputs[1]))
                next_token_logits = outputs[0][0, -1, :] / (temperature if temperature > 0 else 1.)
                # print('single 1', next_token_logits)
                # print(next_token_logits.shape)
                # for _ in set(generated.view(-1).tolist()):
                #     next_token_logits[_] /= repetition_penalty
                filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
                # print('single 2', next_token_logits)
                if temperature == 0:  # greedy sampling:
                    # next_token = torch.argmax(filtered_logits).unsqueeze(0)
                    while True:
                        next_token = torch.argmax(filtered_logits).unsqueeze(0)
                        decode_token = self.tokenizer.decode(next_token, clean_up_tokenization_spaces=True,
                                                             skip_special_tokens=True)
                        # print('||{}||'.format(decode_token))
                        if normalize_text(decode_token) in stops:
                            filtered_logits[next_token] = -float('Inf')
                        else:
                            break
                else:
                    next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
                # print('single', next_token)
                generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)
        return generated

    def sample_sequence_batch(self, length: int, context_input: Tensor, context_attn_mask: Tensor,
                              num_samples=1, temperature=1, top_k=0, top_p=0.0, repetition_penalty=1.0):
        assert context_input.dim() == 2
        context_ids = context_input.unsqueeze(1).repeat(1, num_samples, 1).view(-1, context_input.shape[1])
        context_mask = context_attn_mask.unsqueeze(1).repeat(1, num_samples, 1).view(-1, context_input.shape[1])
        generated = context_ids
        generated_mask = context_mask

        with torch.no_grad():
            for _ in range(length):
                inputs = {'input_ids': generated, 'attention_mask': generated_mask}

                outputs = self.model(**inputs)
                next_token_logits = outputs[0][:, -1, :] / (temperature if temperature > 0 else 1.)
                # print('batch 1', next_token_logits)
                # for _ in set(generated.view(-1).tolist()):
                #     next_token_logits[:, _] /= repetition_penalty
                filtered_logits = top_k_top_p_filtering_batch(next_token_logits, top_k=top_k, top_p=top_p)
                # print('batch 2', next_token_logits)
                if temperature == 0:  # greedy sampling:
                    while True:
                        next_token = torch.argmax(filtered_logits, dim=-1)
                        decode_token = self.tokenizer.decode(next_token, clean_up_tokenization_spaces=True,
                                                             skip_special_tokens=True)
                        # print('||{}||'.format(decode_token))
                        if normalize_text(decode_token) in stops:
                            filtered_logits[0][next_token] = -float('Inf')
                        else:
                            break
                    next_token = next_token.unsqueeze(-1)
                    # next_token = torch.argmax(filtered_logits, dim=-1).unsqueeze(-1)
                else:
                    next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
                # print('batch', next_token)
                # print(next_token.shape)
                generated = torch.cat((generated, next_token), dim=1)
                next_attn_mask = torch.ones_like(next_token, dtype=torch.long, device=self.config.device)
                generated_mask = torch.cat((generated_mask, next_attn_mask), dim=1)
        return generated

    def predict_next_sentence(self, context: str):
        context_tokens_inputs = self.tokenizer([context], return_tensors="pt", padding=True)
        # print('single', context_tokens_inputs)
        for key in context_tokens_inputs.keys():
            context_tokens_inputs[key] = context_tokens_inputs[key].to(self.config.device)
        out = self.sample_sequence_single(
            context_input=context_tokens_inputs['input_ids'],
            length=self.config.length,
            temperature=self.config.temperature,
            top_k=self.config.top_k,
            top_p=self.config.top_p,
            repetition_penalty=self.config.repetition_penalty,
        )
        out = out[0, context_tokens_inputs['input_ids'].shape[1]:].tolist()
        text = self.tokenizer.decode(out, clean_up_tokenization_spaces=True, skip_special_tokens=True)
        text = text[: text.find(self.config.stop_token) if self.config.stop_token else None]
        return text

    def predict_next_word(self, context):
        next_sentence = self.predict_next_sentence(context=context)
        # print(next_sentence)
        next_word = self.first_word_extraction(sentence=next_sentence)
        return next_word

    def first_word_extraction(self, sentence):
        if len(sentence) == 0:
            return ''
        sent_words = sentence.split()
        assert len(sent_words) > 0, 'sentence = {}'.format(sentence)
        sent_words = [normalize_text(text=_) for _ in sent_words]
        next_word = sent_words[0]
        for _ in sent_words:
            if (len(_) > 0) and (_ not in stops):
            # if len(_) > 0:
                next_word = _
                break
        # print('generate', (sentence, next_word))
        return next_word

    def batch_predict_next_sentence(self, context_list: list):
        context_tokens_inputs = self.tokenizer(context_list, return_tensors="pt", padding=True)
        # print('batch', context_tokens_inputs)
        for key in context_tokens_inputs.keys():
            context_tokens_inputs[key] = context_tokens_inputs[key].to(self.config.device)
        out = self.sample_sequence_batch(length=self.config.length,
                                         context_input=context_tokens_inputs["input_ids"],
                                         context_attn_mask=context_tokens_inputs["attention_mask"],
                                         temperature=self.config.temperature,
                                         top_k=self.config.top_k,
                                         top_p=self.config.top_p,
                                         repetition_penalty=self.config.repetition_penalty)
        out = out[:, context_tokens_inputs['input_ids'].shape[1]:].tolist()
        text = [self.tokenizer.decode(_, clean_up_tokenization_spaces=True, skip_special_tokens=True) for _ in out]
        return text

    def batch_predict_next_word(self, context_list: list):
        batch_sentences = self.batch_predict_next_sentence(context_list=context_list)
        # print(batch_sentences)
        batch_next_words = []
        for sentence in batch_sentences:
            # print(sentence)
            next_word = self.first_word_extraction(sentence=sentence)
            batch_next_words.append(next_word)
        return batch_next_words


if __name__ == '__main__':
    from text_generation_codes.argument_lambada_parser import default_argument_parser, completed_argument_parser

    args = default_argument_parser()
    args = completed_argument_parser(args=args)
    args.length = 5
    prompt = "give me a minute to change and i'll meet you at the docks.'' she'd forced those words through her teeth. ``no need to change. we won't be that long.'' shane gripped her arm and started leading her to the dock. ``i can make it there on my own,"
    prompts = ["I love to shake"]
    next_predictor = NextWordPrediction(config=args)
    print(next_predictor.tokenizer.bos_token)
    # # word = next_predictor.predict_next_word(context=prompt)
    # # print('Next word: {}'.format(word))
    # words = next_predictor.batch_predict_next_word(context_list=prompts)
    # print(words)
    # next_predictor.generate_next_word(context=prompt + ' ' + prompt + ' ' + prompt)
#     # next_predictor.generate_next_word(context=prompt)
#     # x = torch.rand((2,5))
#     # sorted_logits, sorted_indices = torch.sort(x, dim=-1, descending=True)
#     # cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
#     # print(cumulative_probs)
#     # sorted_indices_to_remove = cumulative_probs > 0.7
#     # print(sorted_indices_to_remove)
#     # sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
#     # sorted_indices_to_remove[..., 0] = 0
#     # print(sorted_indices_to_remove)
#     # sorted_ids = sorted_indices_to_remove.gather(1, sorted_indices)
#     # print(sorted_ids)
#     #
#     # print('*' * 10)
#     # print(sorted_indices)
#     # indices_to_remove = sorted_indices[sorted_indices_to_remove]
#     # print(indices_to_remove)

