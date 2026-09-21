"""
N-gram language models (unigram / bigram / trigram) with add-k smoothing.
- collecting counts from text tokenized with tokenizer (unigram, bigram and trigram counts)
  stored based on the index into the vocabulary".
    - we are counting over integer token ids returned by BPETokenizer.encode.
- inference consists of returning the negative log probability of a sequence
    - sequence_neglogprob().
- use add-k smoothing (i.e, add-1 smoothing if k=1)".
    - configurable `k`
- perplexity as the per-word average negative log probability of a test set
    - perplexity() and cross_entropy

we are not using any library that implements N-gram models.
only plain dictionaries and the standard library `math` module are used.
"""

import math


class NGramLM:
    """A single N-gram model with add-k (Laplace-family) smoothing.
    Probability of a token given its (n-1)-token context:
        P(w_i | context) = (count(context, w_i) + k) / (count(context) + k * V)
    V = vocabulary size.
    In the unigram case (n=1) the "context" is
    empty and count(context) is just the total number of tokens N:
        P(w) = (count(w) + k) / (N + k * V)
    Add-k reserves probability mass for unseen n-grams so the model never assigns
    zero probability
    """

    def __init__(self, n, vocab_size, k=1.0, bos_id=1, eos_id=2):
        assert n >= 1, "n must be >= 1"
        self.n = n
        self.vocab_total = vocab_size # the "V" in the formula above
        self.k = k
        self.bos_id = bos_id
        self.eos_id = eos_id
        # How many times we saw each full n-gram, and how many times we saw each
        # (n-1)-token context on its own. Keeping the context numbers in their own
        # dictionary means we can look up the bottom of the fraction straight away
        # instead of adding things up again every time.
        self.ngram_tally = {}
        self.context_tally = {}
        self.tokens_seen = 0 # the "N" used by the unigram bottom line

    #  padding
    def _add_markers(self, token_ids):
        """add (n-1) <s> markers before the sequence and one </s> after.
        padding with n-1 BOS symbols means every real token has a full-length
        context, and the trailing EOS lets the model learn how sentences end.
        """
        front_markers = [self.bos_id] * (self.n - 1)
        middle = list(token_ids)
        marked = front_markers + middle + [self.eos_id]
        return marked

    # training
    def train(self, sequences):
        """Accumulate n-gram and context counts from tokenized sequences.
        `sequences` is an iterable of token-id lists (e.g. one per line/sentence).
        """
        for one_seq in sequences:
            marked = self._add_markers(one_seq)
            # every real token plus the </s> after it counts as a prediction.
            self.tokens_seen = self.tokens_seen + len(one_seq) + 1
            # slide a window of width n along the padded sequence.
            for last in range(self.n - 1, len(marked)):
                first = last - self.n + 1
                window = tuple(marked[first:last + 1])
                # add one to this window's tally (starting from 0 if it is new).
                self.ngram_tally[window] = self.ngram_tally.get(window, 0) + 1
                if self.n > 1:
                    # the context is the window without its final token.
                    context = window[:-1]
                    self.context_tally[context] = self.context_tally.get(context, 0) + 1
        return self

    # inference
    def _window_logprob(self, window):
        """log P(last id | preceding ids) under add-k smoothing (natural log)."""
        # how often did we see this exact window? 0 if we never did.
        seen_window = self.ngram_tally.get(window, 0)
        if self.n == 1:
            # unigram: there is no context, so the bottom is every token we saw.
            top = seen_window + self.k
            bottom = self.tokens_seen + self.k * self.vocab_total
        else:
            context = window[:-1]
            seen_context = self.context_tally.get(context, 0)
            top = seen_window + self.k
            bottom = seen_context + self.k * self.vocab_total

        return math.log(top) - math.log(bottom)

    def sequence_neglogprob(self, token_ids):
        """return the negative log probability of a whole sequence (natural log).
        lower is better fit
        """
        marked = self._add_markers(token_ids)
        running_total = 0.0
        for last in range(self.n - 1, len(marked)):
            first = last - self.n + 1
            window = tuple(marked[first:last + 1])
            running_total = running_total - self._window_logprob(window)

        return running_total

    def _score_whole_corpus(self, sequences):
        """total NLL and number of predicted tokens over a corpus."""
        score_total = 0.0
        token_count = 0
        for one_seq in sequences:
            score_total = score_total + self.sequence_neglogprob(one_seq)
            token_count = token_count + len(one_seq) + 1 # real tokens + </s>
        return score_total, token_count

    def cross_entropy(self, sequences):
        """Per-token average negative log probability (nats).
        Perplexity is its exponential.
        """
        score_total, token_count = self._score_whole_corpus(sequences)
        # Never divide by zero, even if the corpus turned out to be empty.
        if token_count < 1:
            token_count = 1

        return score_total / token_count

    def perplexity(self, sequences):
        """Standard perplexity = exp(mean per-token NLL).
        we're using natural log throughout, PP = exp(cross_entropy). (PP = 2 for log base 2**
        (cross_entropy_bits); the value is identical because the base cancels between
        the exponent and the log.)
        """
        # Turn it into a list first, in case we were handed a generator that can
        # only be walked through once.
        seq_list = list(sequences)
        return math.exp(self.cross_entropy(seq_list))

def build_lm(n, tokenizer, text, k=1.0):
    """Convenience: tokenize `text` line-by-line and train an n-gram LM on it.
    splitting on lines gives us natural "sentence-ish" units to pad with
    <s>/</s>. Empty lines are skipped.
    """
    all_seqs = []
    for line in text.splitlines():
        if line.strip() != "":
            all_seqs.append(tokenizer.encode(line))

    model = NGramLM(n=n, vocab_size=tokenizer.actual_vocab_size, k=k,
                    bos_id=tokenizer.bos_id, eos_id=tokenizer.eos_id)
    model.train(all_seqs)
    return model
