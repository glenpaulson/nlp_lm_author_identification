"""
LM-based author identification.

approach we're following: 
  * train 1 shared BPE tokenizer on the combined training text of both authors
    so that token ids mean the same thing for both models and out-of-vocabulary
    handling is symmetric.
  * train 1 N-gram LM per author (Tolkien on The Hobbit, Conan-Doyle on The
    Lost World) using that shared tokenizer.
  * to classify an unseen passage, score it under each author's LM and pick the
    author whose model gives the *lower* negative log probability (= higher
    probability). We length-normalize by using per-token average NLL so the
    decision does not depend on how long the passage is.

The class below bundles the tokenizer + the two LMs and the hyper-parameters
(vocab size, n, k, pre-tokenization) so an experiment can sweep them and the
final `classify_file` entry point can commit to the best-performing choice.
"""

import os

from .bpe_tokenizer import BPETokenizer
from .ngram_lm import NGramLM


class AuthorIdentifier:

    def __init__(self, vocab_size=5000, n=3, k=0.01, pre_tokenizer="whitespace"):
        # settings. Nothing is trained until fit() is called.
        self.vocab_size = vocab_size
        self.n = n
        self.k = k
        self.pre_tok_name = pre_tokenizer
        self.shared_tokenizer = None # 1 tokenizer for every author
        self.author_models = {} # {author label: their trained NGramLM}

    def fit(self, author_texts, train_files):
        """train the shared tokenizer and 1 LM per author.
        parameters:
        author_texts : {author_label: training_text}
        train_files  : the file paths the tokenizer is trained on (the shared vocabulary is learned from both authors' text combined)
        """
        # Step 1: one tokenizer, trained on both books together.
        tokenizer = BPETokenizer(vocab_size=self.vocab_size, pre_tokenizer=self.pre_tok_name)
        tokenizer.train(train_files)
        self.shared_tokenizer = tokenizer
        # Step 2: one language model per author, all sharing those token ids.
        for writer in author_texts:
            training_text = author_texts[writer]
            # turn each non-blank line into a list of token ids.
            line_ids = []
            for one_line in training_text.splitlines():
                if one_line.strip() != "":
                    line_ids.append(tokenizer.encode(one_line))
            model = NGramLM(n=self.n, vocab_size=tokenizer.actual_vocab_size, k=self.k, bos_id=tokenizer.bos_id, eos_id=tokenizer.eos_id)
            model.train(line_ids)
            self.author_models[writer] = model

        return self

    def score(self, passage):
        """Per-token average NLL of `passage` under each author's LM where lower=better fit"""
        assert self.shared_tokenizer is not None, "call fit() first"
        passage_ids = self.shared_tokenizer.encode(passage)
        score_per_author = {}
        for writer in self.author_models:
            model = self.author_models[writer]
            # length-normalized so long/short passages compare fairly.
            score_per_author[writer] = model.cross_entropy([passage_ids])
        return score_per_author

    def predict(self, passage):
        """return the most likely author label for a passage."""
        score_per_author = self.score(passage)
        # walk through the authors and remember the one with the lowest score.
        # lowest NLL wins. If two authors somehow tie, the first one stays.
        best_writer = None
        best_score = None
        for writer in score_per_author:
            this_score = score_per_author[writer]
            if best_score is None or this_score < best_score:
                best_writer = writer
                best_score = this_score

        return best_writer

    def classify_file(self, path):
        """classify a test file where each non-empty line is treated as one passage.

        returns a list of (passage, predicted_author). We keep the passage text
        so the caller can write a readable labels file.
        """
        answers = []
        with open(path, encoding="utf-8", errors="replace") as test_file:
            for raw_line in test_file:
                one_line = raw_line.strip()
                if one_line != "":
                    guess = self.predict(one_line)
                    answers.append((one_line, guess))
        return answers

def load_author_texts(data_dir):
    """load the two training corpora, mapping a clear author label to its text."""
    hobbit_path = os.path.join(data_dir, "hobbit.txt")
    lostworld_path = os.path.join(data_dir, "lostworld.txt")
    with open(hobbit_path, encoding="utf-8") as f:
        hobbit_text = f.read()
    with open(lostworld_path, encoding="utf-8") as f:
        lostworld_text = f.read()
    author_texts = {}
    author_texts["Tolkien"] = hobbit_text
    author_texts["Conan-Doyle"] = lostworld_text
    return author_texts
