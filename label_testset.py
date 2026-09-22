"""
Format the output as per HW2 test set output requirement
Usage:
    python label_testset.py path/to/testset.txt
    python label_testset.py path/to/testset.txt --out outputs/hw2_predictions.txt
"""

import argparse
import io
import os
import re
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from src.author_id import AuthorIdentifier, load_author_texts

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "data")

# final configuration (same as classify_testset.py / evaluate.py).
FINAL_N = 2
FINAL_K = 0.1
FINAL_VOCAB = 2000
FINAL_PRETOK = "whitespace"

OUTPUT_LABEL = {"Tolkien": "Tolkien", "Conan-Doyle": "Doyle"}

def train_final_identifier():
    """Train the final identifier on the full training corpora."""
    identifier = AuthorIdentifier(vocab_size=FINAL_VOCAB, n=FINAL_N, k=FINAL_K, pre_tokenizer=FINAL_PRETOK)
    train_files = [os.path.join(DATA_DIR, "hobbit.txt"),os.path.join(DATA_DIR, "lostworld.txt")]
    identifier.fit(load_author_texts(DATA_DIR), train_files)
    return identifier

def main():
    default_out = os.path.join(PROJECT_DIR, "outputs", "hw2_testset_predictions.txt")
    parser = argparse.ArgumentParser(description="Label an ID-prefixed test set by author.")
    parser.add_argument("testset", help="Test file: 'ID<whitespace>text' per line.")
    parser.add_argument("--out", default=default_out, help="Where to write 'ID<TAB>label' lines.")
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    identifier = train_final_identifier()
    results = []
    with open(args.testset, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.strip() == "":
                continue  # skip blank lines 
            # The ID is the first whitespace-delimited token; the rest is the text.
            match = re.match(r"^(\S+)\s+(.*)$", line.rstrip("\n"))
            if not match:
                continue
            item_id, text = match.group(1), match.group(2)
            label = OUTPUT_LABEL[identifier.predict(text)]
            results.append((item_id, label))

    with open(args.out, "w", encoding="utf-8") as fh:
        for item_id, label in results:
            fh.write("%s\t%s\n" % (item_id, label))

    # Console summary.
    counts = {}
    for _id, label in results:
        counts[label] = counts.get(label, 0) + 1
    print("Labelled %d items using n=%d, k=%s, vocab=%d, pre-tok=%s"
          % (len(results), FINAL_N, FINAL_K, FINAL_VOCAB, FINAL_PRETOK))
    for label in sorted(counts):
        print("  %-8s %d" % (label, counts[label]))
    print("Wrote: " + args.out)

if __name__ == "__main__":
    main()
