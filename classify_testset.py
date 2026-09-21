"""
produce author-ID labels for an instructor-provided test set
usage:
    python classify_testset.py path/to/testset.txt
    python classify_testset.py path/to/testset.txt --out outputs/predictions.csv

input format: a plain-text file of test passages. Two layouts are supported via
--mode:
  * line (default): one passage per non-empty line. This matches our
                    generated test set and is the expected format.
  * paragraph: passages may span several wrapped lines and are
               separated by blank lines (like book paragraphs). 
               Each blank-line-separated block becomes one passage, with its
               internal line breaks collapsed to spaces.
use --mode paragraph if a "short passage" in the test set wraps across multiple
lines; otherwise the default is correct.

output: a CSV with columns  index,predicted_author,passage  plus a plain .txt of
just the labels in order, and a summary printed to the console.

final configuration
Chosen from the hyper-parameter sweep in experiments.py. Bigram / trigram and
unigram all reached ~100% held-out accuracy on short passages; we default to a
BIGRAM model because it genuinely uses n-gram context while still achieving 
perfect held-out accuracy.
"""

import argparse
import csv
import io
import os
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from src.author_id import AuthorIdentifier, load_author_texts

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "data")

# Final hyper-parameters (see report / experiments.py sweep).
FINAL_N = 2 # bigram
FINAL_K = 0.1 # add-k smoothing constant
FINAL_VOCAB = 2000 # BPE vocabulary size
FINAL_PRETOK = "whitespace"

def read_passages(path, mode):
    """Read the test file into a list of passage strings, honouring `mode`.

    line : every non-empty line is one passage (our default/expected format).
    paragraph : blocks separated by blank lines are passages; a passage that was
                wrapped across several lines is rejoined with spaces.
    """
    with open(path, encoding="utf-8", errors="replace") as fh:
        raw = fh.read()
    passages = []
    if mode == "paragraph":
        # Split on one-or-more blank lines, then flatten each block to one line.
        for block in raw.replace("\r\n", "\n").split("\n\n"):
            joined = " ".join(part.strip() for part in block.splitlines() if part.strip() != "")
            if joined != "":
                passages.append(joined)
    else:  # "line"
        for line in raw.splitlines():
            if line.strip() != "":
                passages.append(line.strip())
    return passages


def train_final_identifier():
    """Train the final identifier on the full training corpora."""
    texts_by_author = load_author_texts(DATA_DIR)
    train_files = [os.path.join(DATA_DIR, "hobbit.txt"),
                   os.path.join(DATA_DIR, "lostworld.txt")]
    identifier = AuthorIdentifier(vocab_size=FINAL_VOCAB,n=FINAL_N,k=FINAL_K, pre_tokenizer=FINAL_PRETOK)
    identifier.fit(texts_by_author, train_files)
    return identifier

def main():
    # Read the command line.
    default_csv = os.path.join(PROJECT_DIR, "outputs", "author_id_predictions.csv")
    parser = argparse.ArgumentParser(description="Classify test passages by author.")
    parser.add_argument("testset",help="Path to the test-set text file (one passage per line).")
    parser.add_argument("--out", default=default_csv,help="Where to write the CSV of predictions.")
    parser.add_argument("--gold", default=None,
                        help="Optional answer-key file (one true author label per line, "
                             "same order as the test set). If given, prints accuracy.")
    parser.add_argument("--mode", default="line", choices=["line", "paragraph"],
                        help="How to split the test file into passages: 'line' (default, "
                             "one passage per line) or 'paragraph' (blank-line-separated "
                             "blocks, for passages that wrap across lines).")
    cli_args = parser.parse_args()
    csv_path = cli_args.out
    out_folder = os.path.dirname(csv_path)
    os.makedirs(out_folder, exist_ok=True)
    # Train, then label every passage in the test file (split per --mode).
    identifier = train_final_identifier()
    passages = read_passages(cli_args.testset, cli_args.mode)
    predictions = [(passage, identifier.predict(passage)) for passage in passages]
    # Write the CSV: one row per passage, numbered from 0.
    with open(csv_path, "w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["index", "predicted_author", "passage"])
        row_number = 0
        for passage, author in predictions:
            writer.writerow([row_number, author, passage])
            row_number = row_number + 1
    # Write the bare labels file next to it, e.g. foo.csv -> foo_labels.txt
    labels_path = os.path.splitext(csv_path)[0] + "_labels.txt"
    with open(labels_path, "w", encoding="utf-8") as labels_file:
        for passage, author in predictions:
            labels_file.write(author + "\n")
    # Count how many passages went to each author.
    how_many = {}
    for passage, author in predictions:
        how_many[author] = how_many.get(author, 0) + 1
    # Put the authors in order, biggest count first. If two authors tie, the one
    # we came across first stays first.
    still_to_place = list(how_many.keys())
    ordered_authors = []
    while len(still_to_place) > 0:
        winner = still_to_place[0]
        for name in still_to_place:
            if how_many[name] > how_many[winner]:
                winner = name
        ordered_authors.append(winner)
        still_to_place.remove(winner)
    # Console summary.
    print("Classified {} passages using n={}, k={}, vocab={}, pre-tok={}".format(len(predictions), FINAL_N, FINAL_K, FINAL_VOCAB, FINAL_PRETOK))
    for author in ordered_authors:
        print("  {:<14} {}".format(author, how_many[author]))
    print("Wrote: " + csv_path)
    print("Wrote: " + labels_path)

    # Optional scoring: if the caller supplied an answer key, report accuracy.
    if cli_args.gold is not None:
        with open(cli_args.gold, encoding="utf-8") as gold_file:
            gold_labels = [ln.strip() for ln in gold_file if ln.strip() != ""]
        predicted_labels = [author for _passage, author in predictions]
        if len(gold_labels) != len(predicted_labels):
            print("WARNING: {} gold labels but {} predictions - cannot score. "
                  "Make sure the gold file has one label per test passage, same order."
                  .format(len(gold_labels), len(predicted_labels)))
        else:
            correct = sum(1 for g, p in zip(gold_labels, predicted_labels) if g == p)
            total = len(gold_labels)
            print("Accuracy vs gold: {}/{} = {:.3f}".format(correct, total, correct / total))

if __name__ == "__main__":
    main()
