"""
run all explorations and print a results table.

This script produces the concrete numbers cited in the report:
1. Bigram vs trigram perplexity on held-out text.
2. Effect of the add-k smoothing constant k on perplexity.
3. Effect of vocabulary size and pre-tokenization strategy.
4. Author-identification accuracy on a held-out labelled pseudo test-set,
   with a small hyper-parameter sweep to choose the final configuration.

Run:  python experiments.py
(Results are also written to outputs/experiment_results.txt)
"""

import io
import os
import sys

# Make stdout UTF-8 so byte-level tokens (e.g. the 'Ġ' space marker) never crash printing on a Windows cp1252 console.
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from src.bpe_tokenizer import BPETokenizer
from src.ngram_lm import NGramLM
from src.author_id import AuthorIdentifier, load_author_texts
from src.utils import split_lines_train_test, make_passages

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
OUT_DIR = os.path.join(PROJECT_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# Every line we print also gets kept here, so we can save the whole report at the end without building it twice.
report_lines = []

def show(line=""):
    """Print one line to the screen and remember it for the report file."""
    print(line)
    report_lines.append(line)

def text_to_id_lists(tokenizer, text):
    """Turn every non-blank line of `text` into its own list of token ids."""
    id_lists = []
    for one_line in text.splitlines():
        if one_line.strip() != "":
            id_lists.append(tokenizer.encode(one_line))
    return id_lists

def run_perplexity_tests(texts_by_author):
    show("=" * 70)
    show("Part A/B: N-gram perplexity - bigram vs trigram, and add-k sweep")
    show("=" * 70)
    show("Perplexity on held-out 10% of each book. Tokenizer + LM trained on the")
    show("other 90%. Lower perplexity = better next-token prediction.\n")

    model_sizes = [(1, "unigram"), (2, "bigram"), (3, "trigram")]
    k_values = [1.0, 0.1, 0.01, 0.001]

    for writer in texts_by_author:
        whole_text = texts_by_author[writer]
        train_text, test_text = split_lines_train_test(whole_text, test_frac=0.1)
        # Train tokenizer on the TRAIN split only (no test leakage). Fixed vocab.
        tokenizer = BPETokenizer(vocab_size=5000, pre_tokenizer="whitespace")
        tokenizer.train_from_text([train_text])
        train_seqs = text_to_id_lists(tokenizer, train_text)
        test_seqs = text_to_id_lists(tokenizer, test_text)
        show("--- {}  (vocab={}, whitespace pre-tok) ---".format(writer, tokenizer.actual_vocab_size))
        show("{:<9}{:>8}   perplexity".format("model", "k"))

        for n, model_name in model_sizes:
            for k in k_values:
                model = NGramLM(n=n, vocab_size=tokenizer.actual_vocab_size, k=k, bos_id=tokenizer.bos_id, eos_id=tokenizer.eos_id)
                model.train(train_seqs)
                pplx = model.perplexity(test_seqs)
                show("{:<9}{:>8}   {:10.2f}".format(model_name, k, pplx))
        show("")

def run_tokenizer_tests(texts_by_author):
    show("=" * 70)
    show("Part A: Tokenizer Exploration, vocab size + pre-tokenization")
    show("=" * 70)
    show("Trigram perplexity (k=0.01) on held-out 10% of The Hobbit for each")
    show("tokenizer setting. Also reports avg tokens/word (fertility).\n")

    hobbit_text = texts_by_author["Tolkien"]
    train_text, test_text = split_lines_train_test(hobbit_text, test_frac=0.1)
    word_count = len(test_text.split())
    pre_tok_choices = ["whitespace", "whitespace_lower", "bytelevel"]
    vocab_choices = [1000, 3000, 5000, 10000]

    show("{:<18}{:>7}{:>8}{:>10}{:>13}".format(
        "pre-tok", "vocab", "actual", "tok/word", "trigram PP"))

    for pre_tok in pre_tok_choices:
        for vocab in vocab_choices:
            tokenizer = BPETokenizer(vocab_size=vocab, pre_tokenizer=pre_tok)
            tokenizer.train_from_text([train_text])
            train_seqs = text_to_id_lists(tokenizer, train_text)
            test_seqs = text_to_id_lists(tokenizer, test_text)
            # How many tokens did the test split turn into altogether?
            token_count = 0
            for one_seq in test_seqs:
                token_count = token_count + len(one_seq)
            model = NGramLM(n=3,vocab_size=tokenizer.actual_vocab_size,k=0.01,bos_id=tokenizer.bos_id,eos_id=tokenizer.eos_id)
            model.train(train_seqs)
            pplx = model.perplexity(test_seqs)
            tokens_per_word = token_count / word_count
            show("{:<18}{:>7}{:>8}{:>10.2f}{:>13.2f}".format(pre_tok, vocab, tokenizer.actual_vocab_size, tokens_per_word, pplx))

    show("\nNote: perplexity is only comparable at a *fixed* vocabulary/tokenization,")
    show("because changing the vocab changes what a 'token' is. We read these as")
    show("trends, and rely on author-ID accuracy mentioned below for the final choice.\n")

def run_author_id_tests(texts_by_author):
    show("=" * 70)
    show("Part C: AUTHOR IDENTIFICATION - held-out accuracy + hyper-param sweep")
    show("=" * 70)
    show("Hold out 12% of each book as short ~40-word passages, the labelled test")
    show("set), train the identifier on the rest, and measure classification")
    show("accuracy. This tells us which (n, k, vocab) to commit to.\n") 
    # Build labelled held-out passages, and the remaining training text per author.
    training_by_author = {}
    labelled_passages = []
    for writer in texts_by_author:
        whole_text = texts_by_author[writer]
        train_text, held_text = split_lines_train_test(whole_text, test_frac=0.12)
        training_by_author[writer] = train_text
        chunks = make_passages(held_text, words_per_passage=40, max_passages=50)
        for one_chunk in chunks:
            labelled_passages.append((one_chunk, writer))
    train_files = [os.path.join(DATA_DIR, "hobbit.txt"),os.path.join(DATA_DIR, "lostworld.txt")]
    # Count how many passages came from each author.
    tolkien_count = 0
    doyle_count = 0
    for passage, gold in labelled_passages:
        if gold == "Tolkien":
            tolkien_count = tolkien_count + 1
        if gold == "Conan-Doyle":
            doyle_count = doyle_count + 1

    show("Labelled test passages: {} ({} Tolkien / {} Conan-Doyle)\n".format(len(labelled_passages), tolkien_count, doyle_count))
    show("{:>3}{:>8}{:>8}{:>18}{:>11}".format("n", "k", "vocab", "pre-tok", "accuracy"))  
    # Remember the settings that scored best. None means that nothing has been tried yet
    best_accuracy = None
    best_n = None
    best_k = None
    best_vocab = None
    best_pre_tok = None

    for n in (1, 2, 3):
        for k in (1.0, 0.1, 0.01):
            for vocab in (2000, 5000):
                pre_tok = "whitespace"
                identifier = AuthorIdentifier(vocab_size=vocab, n=n, k=k,pre_tokenizer=pre_tok)
                # Fit LMs on the held-out-removed training text; tokenizer on full files.
                identifier.fit(training_by_author, train_files)
                # Count how many passages it got right.
                correct = 0
                for passage, gold in labelled_passages:
                    if identifier.predict(passage) == gold:
                        correct = correct + 1
                accuracy = correct / len(labelled_passages)
                show("{:>3}{:>8}{:>8}{:>18}{:>11.3f}".format(n, k, vocab, pre_tok, accuracy))

                if best_accuracy is None or accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_n = n
                    best_k = k
                    best_vocab = vocab
                    best_pre_tok = pre_tok

    show("")
    show("Best: accuracy={:.3f} with n={}, k={}, vocab={}, pre-tok={}".format(best_accuracy, best_n, best_k, best_vocab, best_pre_tok))

    return (best_accuracy, best_n, best_k, best_vocab, best_pre_tok)

def main():
    texts_by_author = load_author_texts(DATA_DIR)
    run_perplexity_tests(texts_by_author)
    run_tokenizer_tests(texts_by_author)
    best = run_author_id_tests(texts_by_author)
    # Save everything we printed into one report file.
    results_path = os.path.join(OUT_DIR, "experiment_results.txt")
    with open(results_path, "w", encoding="utf-8") as out_file:
        out_file.write("\n".join(report_lines) + "\n")

    show("\nsaved the results to outputs/experiment_results.txt")
    return best

if __name__ == "__main__":
    main()
