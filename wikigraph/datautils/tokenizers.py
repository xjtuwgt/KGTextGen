"""Tokenizers for text data."""
import abc
import csv
import io
import re
from typing import List

import nltk
import numpy as np
from wikigraph.datautils import io_utils


class Tokenizer(abc.ABC):
    """Base class for tokenizers."""
    @abc.abstractmethod
    def encode(self,
               inputs: str,
               prepend_bos: bool = False,
               append_eos: bool = False) -> np.ndarray:
        """Encode input string into an array of token IDs.
        Args:
          inputs: a string.
          prepend_bos: set to True to add <bos> token at the beginning of the token
            sequence.
          append_eos: set to True to add <eos> token at the end of the token
            sequence.
        Returns:
          tokens: [n_tokens] int array.
        """

    @abc.abstractmethod
    def decode(self, inputs) -> str:
        """Decode a sequence of tokens back into a string.
        Args:
          inputs: array or list of ints.
        Returns:
          s: the decoded string using this tokenizer.
        """

    @property
    @abc.abstractmethod
    def vocab_size(self) -> int:
        """Size of the vocabulary."""

    @abc.abstractmethod
    def pad_token(self) -> int:
        """ID of the <pad> token."""

    @abc.abstractmethod
    def bos_token(self) -> int:
        """ID of the <bos> token."""


class WordTokenizer(Tokenizer):
    """Word-level tokenizer for white-space separated text data."""
    def __init__(self, vocab_file: str):
        """Constructor.
        Args:
          vocab_file: a csv vocab file.
        """
        content = io_utils.read_cvs_file(vocab_file, encoding='utf-8')
        vocab = [w for w, _ in content]

        # Add pad and bos tokens to the vocab
        to_add = ['<pad>', '<bos>']
        if '<unk>' not in vocab:
            to_add.append('<unk>')
        vocab = to_add + vocab

        # token-index mappings
        self._t2i = {t: i for i, t in enumerate(vocab)}
        self._i2t = {i: t for t, i in self._t2i.items()}

        self._unk_token = self._t2i['<unk>']
        self._bos_token = self._t2i['<bos>']
        self._pad_token = self._t2i['<pad>']

    @property
    def vocab_size(self):
        return len(self._t2i)

    def encode(self, inputs, prepend_bos=False, append_eos=False):
        tokens = [self._t2i.get(t, self._unk_token) for t in inputs.split(' ') if t]
        if prepend_bos:
            tokens = [self._bos_token] + tokens
        if append_eos:
            # Reuse <bos> as <eos>.
            tokens.append(self._bos_token)
        return np.array(tokens, dtype=np.int32)

    def decode(self, inputs):
        """Decode a sequence of token IDs back into a string."""
        # Remove the first <bos> token if there is any.
        if inputs[0] == self._bos_token:
            inputs = inputs[1:]
        tokens = []
        for i in inputs:
            # Use <bos> also as <eos> and stop there.
            if i == self._bos_token:
              break
            tokens.append(self._i2t[i])
        return ' '.join(tokens)

    def pad_token(self):
        return self._pad_token

    def bos_token(self):
        return self._bos_token


class GraphTokenizer:
    """Tokenizer for the content on the graphs."""
    def __init__(self, vocab_file: str):
        """Constructor.
        Args:
          vocab_file: path to a vocab file.
        """
        # content = io_utils.read_txt_file(vocab_file, encoding='utf-16')
        # vocab = content.split('\n')
        content = io_utils.read_cvs_file(vocab_file, encoding='utf-8')
        vocab = [w for w, _ in content]

        vocab = ['<pad>', '<bos>', '<unk>'] + vocab

        # token-index mappings
        self._t2i = {t: i for i, t in enumerate(vocab)}
        self._i2t = {i: t for t, i in self._t2i.items()}

        self._unk_token = self._t2i['<unk>']
        self._bos_token = self._t2i['<bos>']
        self._pad_token = self._t2i['<pad>']

    @property
    def vocab_size(self):
        return len(self._t2i)

    def encode_node(self, txt: str) -> np.ndarray:
        return np.array([self._t2i.get(t, self._unk_token)
                         for t in self.split_node(txt)])

    def encode_edge(self, txt: str) -> np.ndarray:
        return np.array([self._t2i.get(t, self._unk_token)
                         for t in self.split_edge(txt)])

    def encode(self, inputs, prepend_bos=False, append_eos=False):
        tokens = [self._t2i.get(t, self._unk_token) for t in inputs.split(' ') if t]
        if prepend_bos:
          tokens = [self._bos_token] + tokens
        if append_eos:
          # Reuse <bos> as <eos>.
          tokens.append(self._bos_token)
        return np.array(tokens, dtype=np.int32)

    def decode(self, inputs):
        """Decode a sequence of token IDs back into a string."""
        # Remove the first <bos> token if there is any.
        if inputs[0] == self._bos_token:
            inputs = inputs[1:]
        tokens = []
        for i in inputs:
            # Use <bos> also as <eos> and stop there.
            if i == self._bos_token:
                break
            tokens.append(self._i2t[i])
        return ' '.join(tokens)

    @classmethod
    def split_node(cls, txt: str) -> List[str]:
        """Split a node string into a sequence of tokens."""
        if txt[0] == '"' and txt[-1] == '"':  # Node is a string literal.
            norm_txt = io_utils.normalize_freebase_string(txt[1:-1].lower())
            tokens = nltk.wordpunct_tokenize(norm_txt)
            for i, t in enumerate(tokens):
                if t.isnumeric():
                    tokens[i] = '<number>'
            return tokens
        else:  # If node is not a string literal it is always an entity.
            return ['<entity>']

    @classmethod
    def split_edge(cls, txt: str) -> List[str]:
        """Split an edge string into a sequence of tokens."""
        tokens = re.split('[._ ]+', txt.lower().split('/')[1])
        return tokens

    def pad_token(self):
        return self._pad_token

    def bos_token(self):
        return self._bos_token