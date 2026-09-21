"""
generate our OWN labelled test set and evaluate the author identifier on it.

Why this file exists
Deliverable #3 asks for author-ID labels on a test set. The instructor's test
set is not in hand yet, so this script builds an honest stand-in:

  * We hold out a slice of each book (default 15%) that the models never train
    on, cut it into short ~40-word passages, and label each passage with its
    true author. That labelled set is written to  testset/  .
  * We train the author identifier ONLY on the remaining text (and even train
    the shared tokenizer on that remaining text, via temp files, so there is no
    leakage at all).
  * We predict every passage, write the predicted labels, and report accuracy
    plus a confusion matrix.

This gives a real, defensible generalization number and produces a concrete
labels artifact that completes the deliverable as a demonstration. When the
real test set arrives, `classify_testset.py` is the tool to label it (see
TESTING.md); this script is the self-contained evaluation.

Run:  python evaluate.py
"""

import io
import os
import sys
import tempfile

# UTF-8 stdout so any byte-level tokens never crash printing on Windows.
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from src.author_id import AuthorIdentifier, load_author_texts
from src.utils import split_lines_train_test, make_passages

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
TESTSET_DIR = os.path.join(PROJECT_DIR, "testset")
OUT_DIR = os.path.join(PROJECT_DIR, "outputs")

# Final model configuration (matches classify_testset.py).
FINAL_N = 2
FINAL_K = 0.1
FINAL_VOCAB = 2000
FINAL_PRETOK = "whitespace"
# How the held-out test set is built.
TEST_FRAC = 0.15 # fraction of each book held out for testing
WORDS_PER_PASSAGE = 40 # each test passage is ~40 words (short, like the real set)
MAX_PASSAGES_PER_AUTHOR = 60
SEED = 13 # deterministic split -> reproducible test set

def build_heldout_testset():
    """Split each book, return (train_texts, gold) where gold = [(passage, author)]."""
    author_texts = load_author_texts(DATA_DIR)
    train_texts = {}
    gold = []
    for author, text in author_texts.items():
        train_text, held_text = split_lines_train_test(text, test_frac=TEST_FRAC, seed=SEED)
        train_texts[author] = train_text
        for passage in make_passages(held_text, WORDS_PER_PASSAGE, MAX_PASSAGES_PER_AUTHOR):
            gold.append((passage, author))
    return train_texts, gold

def write_testset_files(gold):
    """Write the passages and the answer key so they are reusable."""
    os.makedirs(TESTSET_DIR, exist_ok=True)
    passages_path = os.path.join(TESTSET_DIR, "generated_testset.txt")
    gold_path = os.path.join(TESTSET_DIR, "generated_testset_gold.txt")
    with open(passages_path, "w", encoding="utf-8") as fh:
        for passage, _author in gold:
            fh.write(passage + "\n")
    with open(gold_path, "w", encoding="utf-8") as fh:
        for _passage, author in gold:
            fh.write(author + "\n")
    return passages_path, gold_path

def train_identifier_leakfree(train_texts):
    """Train the identifier on train-only text.
    The shared tokenizer is trained on the train split too (written to temp
    files), so the held-out passages are unseen by every component.
    """
    with tempfile.TemporaryDirectory() as tmp:
        train_files = []
        for author, text in train_texts.items():
            path = os.path.join(tmp, "train_%s.txt" % author.replace("-", "_"))
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(text)
            train_files.append(path)
        identifier = AuthorIdentifier(vocab_size=FINAL_VOCAB, n=FINAL_N, k=FINAL_K, pre_tokenizer=FINAL_PRETOK)
        # fit() reads train_files while the temp dir is still open.
        identifier.fit(train_texts, train_files)
    return identifier

def evaluate(identifier, gold):
    """Predict each passage; return predictions and metrics."""
    authors = sorted({a for _p, a in gold})
    # Confusion counts: confusion[true][pred]
    confusion = {a: {b: 0 for b in authors} for a in authors}
    predictions = []
    correct = 0
    for passage, true_author in gold:
        pred = identifier.predict(passage)
        predictions.append((passage, true_author, pred))
        confusion[true_author][pred] += 1
        if pred == true_author:
            correct += 1
    accuracy = correct / len(gold) if gold else 0.0
    return predictions, accuracy, confusion, authors

def write_outputs(predictions, accuracy, confusion, authors):
    os.makedirs(OUT_DIR, exist_ok=True)
    # Predicted-labels artifact, one label per line, aligned
    # with testset/generated_testset.txt.
    labels_path = os.path.join(OUT_DIR, "generated_predictions_labels.txt")
    with open(labels_path, "w", encoding="utf-8") as fh:
        for _passage, _true, pred in predictions:
            fh.write(pred + "\n")
    # A readable CSV with the gold label alongside, for auditing.
    import csv
    csv_path = os.path.join(OUT_DIR, "generated_predictions.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["index", "true_author", "predicted_author", "correct", "passage"])
        for i, (passage, true, pred) in enumerate(predictions):
            w.writerow([i, true, pred, int(true == pred), passage])

    # Metrics report.
    lines = []
    lines.append("Author-identification evaluation on our held-out test set")
    lines.append("=" * 60)
    lines.append("Config: n=%d, k=%s, vocab=%d, pre-tok=%s" %
                 (FINAL_N, FINAL_K, FINAL_VOCAB, FINAL_PRETOK))
    lines.append("Test passages: %d (held out, never seen in training)" %
                 sum(sum(row.values()) for row in confusion.values()))
    lines.append("Accuracy: %.3f" % accuracy)
    lines.append("")
    lines.append("Confusion matrix (rows = true author, cols = predicted):")
    header = "%-16s" % "" + "".join("%-16s" % a for a in authors)
    lines.append(header)
    for a in authors:
        row = "%-16s" % a + "".join("%-16d" % confusion[a][b] for b in authors)
        lines.append(row)
    report = "\n".join(lines)
    report_path = os.path.join(OUT_DIR, "evaluation_report.txt")
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(report + "\n")
    return labels_path, csv_path, report_path, report

def main():
    train_texts, gold = build_heldout_testset()
    passages_path, gold_path = write_testset_files(gold)
    identifier = train_identifier_leakfree(train_texts)
    predictions, accuracy, confusion, authors = evaluate(identifier, gold)
    labels_path, csv_path, report_path, report = write_outputs(
        predictions, accuracy, confusion, authors)

    print(report)
    print("")
    print("Wrote test set : %s" % os.path.relpath(passages_path, PROJECT_DIR))
    print("Wrote answers : %s" % os.path.relpath(gold_path, PROJECT_DIR))
    print("Wrote labels : %s" % os.path.relpath(labels_path, PROJECT_DIR))
    print("Wrote CSV : %s" % os.path.relpath(csv_path, PROJECT_DIR))
    print("Wrote report : %s" % os.path.relpath(report_path, PROJECT_DIR))

if __name__ == "__main__":
    main()
