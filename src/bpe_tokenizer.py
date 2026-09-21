"""
BPE tokenizer wrapper around the Hugging Face `tokenizers` library.
We are not reimplementing byte-pair-encoding ourselves.
This module is an interface that:
  * trains a BPE vocabulary of a chosen size on a corpus,
  * exposes a small, stable API (encode -> vocab indices, decode, vocab size, special-token ids) that the rest of our code (the N-gram LMs) depends on,
  * helps us to experiment with the vocabulary size and the pre-tokenization strategy.

The N-gram models only ever see integer token ids which is the index into the
vocabulary provided by your tokenizer, which is exactly what `encode` returns.
References consulted: Hugging Face tokenizers quicktour https://huggingface.co/docs/tokenizers/en/quicktour
What's done below -> (models.BPE + a BpeTrainer + a pre-tokenizer) follows
that quicktour; the wrapper class and all comments are my own.
"""

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, normalizers, decoders

# Special tokens. The N-gram models use <s>/</s> as sentence boundary markers and
# <unk> for any token the trained vocabulary cannot represent.
UNK = "<unk>"
BOS = "<s>" # beginning of sentence
EOS = "</s>" # end of sentence
PAD = "<pad>"
SPECIAL_TOKENS = [UNK, BOS, EOS, PAD]

class BPETokenizer:
    """A trainable BPE tokenizer that gives vocabulary indices.
    Parameters:
    vocab_size : target number of merges/tokens in the final vocabulary.
    pre_tokenizer : which pre-tokenization strategy to use before BPE merges:
        - "whitespace" : split on whitespace and punctuation.
        - "bytelevel"  : GPT-2 style byte-level, no <unk> ever (bytes are the base alphabet).
        - "whitespace_lower" : whitespace split + lowercasing normalizer.
    min_frequency : a merge must occur at least this many times to be learned.
    """

    def __init__(self, vocab_size=5000, pre_tokenizer="whitespace", min_frequency=2):
        # Just remember the settings. Nothing is trained yet.
        self.vocab_size = vocab_size
        self.pre_tokenizer_name = pre_tokenizer
        self.min_frequency = min_frequency
        self.tokenizer = None # the real HF Tokenizer goes here once we train

    # build
    def _new_tokenizer(self):
        """Construct an *untrained* HF BPE tokenizer with the chosen pre-tokenizer.
        """
        if self.pre_tokenizer_name == "bytelevel":
            tok = Tokenizer(models.BPE(unk_token=None))
            # add_prefix_space=False keeps the first word from getting a leading space marker; fine for our line-based corpora.
            tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
            tok.decoder = decoders.ByteLevel()
        else:
            tok = Tokenizer(models.BPE(unk_token=UNK))
            if self.pre_tokenizer_name == "whitespace_lower":
                # Normalizers run before pre-tokenization; lowercase folds case so "The" and "the" share a token.
                tok.normalizer = normalizers.Sequence([normalizers.NFD(), normalizers.Lowercase()])
            # Whitespace() splits on whitespace *and* isolates punctuation
            tok.pre_tokenizer = pre_tokenizers.Whitespace()
            tok.decoder = decoders.BPEDecoder()
        return tok

    def _new_trainer(self):
        """build the trainer object that actually learns the merges.
        both train() and train_from_text() need exactly the same trainer.
        """
        # ByteLevel exposes an initial byte alphabet so every byte is representable.
        if self.pre_tokenizer_name == "bytelevel":
            initial_alphabet = pre_tokenizers.ByteLevel.alphabet()
        else:
            initial_alphabet = []
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=self.min_frequency,
            special_tokens=SPECIAL_TOKENS,
            initial_alphabet=initial_alphabet,
            show_progress=False,
        )
        return trainer

    def train(self, files):
        """learn a BPE vocabulary from one or more plain-text files."""
        tok = self._new_tokenizer()
        trainer = self._new_trainer()
        file_list = list(files)
        tok.train(file_list, trainer)
        self.tokenizer = tok
        return self

    def train_from_text(self, texts):
        """learn a BPE vocabulary from in-memory strings
        Used when we must train the tokenizer on a *train split only*, this helps to keep
        the held-out perplexity test set truly unseen.
        """
        tok = self._new_tokenizer()
        trainer = self._new_trainer()
        # Collect every non-blank line from every text into one list.
        lines = []
        for one_text in texts:
            for line in one_text.splitlines():
                if line.strip() != "":
                    lines.append(line)
        tok.train_from_iterator(lines, trainer)
        self.tokenizer = tok
        return self

    # interface
    def _ensure(self):
        """give back the trained tokenizer, or complain if we do not have one yet."""
        if self.tokenizer is None:
            raise RuntimeError("tokenizer is not trained/loaded yet. please call train() or load().")
        return self.tokenizer

    def encode(self, text):
        """return the list of vocabulary indices (token ids) for `text`.
        which are the integer ids the N-gram models count over.
        """
        tok = self._ensure()
        result = tok.encode(text)
        return result.ids

    def encode_tokens(self, text):
        """return the surface tokens (strings)"""
        tok = self._ensure()
        result = tok.encode(text)
        return result.tokens

    def decode(self, ids):
        """Inverse of `encode` - best-effort; special tokens are skipped"""
        tok = self._ensure()
        return tok.decode(list(ids))

    # `@property` lets the rest of the code write `tok.actual_vocab_size`
    # (no brackets) instead of `tok.actual_vocab_size()`.
    @property
    def actual_vocab_size(self):
        """final vocabulary size - may be < requested if the corpus is small"""
        tok = self._ensure()
        return tok.get_vocab_size()

    def token_to_id(self, token):
        """Look up the id number of one token string."""
        tok = self._ensure()
        return tok.token_to_id(token)

    @property
    def bos_id(self):
        return self.token_to_id(BOS)

    @property
    def eos_id(self):
        return self.token_to_id(EOS)

    @property
    def unk_id(self):
        this_id = self.token_to_id(UNK)
        if this_id is None:
            # No <unk> token in the vocabulary, so report -1 instead.
            return -1
        return this_id

    # persistence
    def save(self, path):
        """Write the trained tokenizer out to a JSON file."""
        tok = self._ensure()
        tok.save(path)

    @classmethod
    def load(cls, path, vocab_size=0, pre_tokenizer="whitespace"):
        """Build a BPETokenizer from a file that save() wrote earlier."""
        obj = cls(vocab_size=vocab_size, pre_tokenizer=pre_tokenizer)
        obj.tokenizer = Tokenizer.from_file(path)
        obj.vocab_size = obj.tokenizer.get_vocab_size()
        return obj


if __name__ == "__main__":
    # interface demo
    import os
    import sys
    # The byte-level strategy marks spaces with "Ġ", which the default Windows
    # console encoding (cp1252) cannot print - the prints below would crash with
    # a UnicodeEncodeError. Switch stdout to UTF-8 so those tokens show up, and
    # fall back to "?" for anything the terminal still cannot draw.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    src_folder = os.path.dirname(os.path.abspath(__file__))
    project_folder = os.path.dirname(src_folder)
    files = [os.path.join(project_folder, "data", "hobbit.txt"),os.path.join(project_folder, "data", "lostworld.txt"),]
    strategies = ["whitespace", "bytelevel", "whitespace_lower"]
    sample = "Professor Challenger roared: the plateau was unmistakably prehistoric!"
    for strat in strategies:
        t = BPETokenizer(vocab_size=3000, pre_tokenizer=strat)
        t.train(files)
        first_tokens = t.encode_tokens(sample)[:8]
        first_ids = t.encode(sample)[:8]
        print("[" + strat + "]",
              " vocab=" + str(t.actual_vocab_size),
              " tokens=" + str(first_tokens) + "...",
              " ids[:8]=" + str(first_ids))
