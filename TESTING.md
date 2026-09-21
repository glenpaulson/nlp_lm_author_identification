# Running the Author-Identification Test Set (for TAs / graders)

This document explains how to run our trained author identifier on a test set
and get predicted author labels. Two workflows are covered:

1. label your own test set (the normal grading case).
2. reproduce our held-out evaluation (our self-generated labelled test set).

The classifier decides between two authors: **`Tolkien`** (*The Hobbit*) and
**`Conan-Doyle`** (*The Lost World*).

---

## 0. One-time setup

From the project root (`nlp_lm_author_identification/`):

```bash
pip install -r requirements.txt 
```

Python 3.9+ is assumed. No GPU or internet is needed.

---

## 1. Label your test set

### Input format
A plain-text file, one test passage per non-empty line, UTF-8 encoded. For
example, `my_testset.txt`:

```
In a hole in the ground there lived a hobbit, and the dwarves sought the mountain.
Professor Challenger insisted the plateau still held living dinosaurs.
```

### Command
```bash
python classify_testset.py my_testset.txt
```

This trains the final model on the full training corpora (`data/hobbit.txt`,
`data/lostworld.txt`) and labels every line. It writes two files:

| File | Contents |
|------|----------|
| `outputs/author_id_predictions.csv` | `index, predicted_author, passage` (one row per passage) |
| `outputs/author_id_predictions_labels.txt` | just the predicted labels, one per line, in input order |

You can choose a different output location with `--out`:
```bash
python classify_testset.py my_testset.txt --out outputs/my_predictions.csv
```
(The labels file is written next to it as `my_predictions_labels.txt`.)

### Optional: score against an answer key
If you have the true labels in a file (one label per line, **same order** as the
test passages), pass `--gold` and the script prints accuracy:

```bash
python classify_testset.py my_testset.txt --gold my_answers.txt
```
Output ends with, e.g., `Accuracy vs gold: 47/50 = 0.940`.

### Input shape / passage splitting
Passages are split **one per non-empty line** by default (`--mode line`). This is
the same format as our generated test set, so if your test set matches it, no
flag is needed.

If instead a "short passage" **wraps across several lines** and passages are
separated by blank lines (like book paragraphs), use `--mode paragraph` — each
blank-line-separated block becomes one passage:

```bash
python classify_testset.py my_testset.txt --mode paragraph
```

Verified: 3 one-line passages -> 3 labels (`line`); a 2-passage wrapped file -> 2
labels (`paragraph`).

---

## 2. Reproduce our held-out evaluation

We also ship a self-generated, labelled test set so the result is reproducible
without any external file. `evaluate.py` holds out 15% of each book (never used
in training), cuts it into ~40-word passages, labels them, trains only on the
rest, and reports accuracy:

```bash
python evaluate.py
```

It prints an accuracy + confusion matrix and writes:

| File | Contents |
|------|----------|
| `testset/generated_testset.txt` | the held-out test passages (one per line) |
| `testset/generated_testset_gold.txt` | the true author of each passage (answer key) |
| `outputs/generated_predictions_labels.txt` | our model's predicted labels |
| `outputs/generated_predictions.csv` | per-passage: true vs predicted vs correct |
| `outputs/evaluation_report.txt` | accuracy and confusion matrix |

Current result: **120 held-out passages, accuracy = 1.000** (see
`outputs/evaluation_report.txt`).

You can also feed our generated test set through the grading tool from step 1:
```bash
python classify_testset.py testset/generated_testset.txt --gold testset/generated_testset_gold.txt
```

---

## 3. Reproduce the exploration tables (optional)

To regenerate every table in the report (perplexity: bigram vs trigram, add-k
sweep, tokenizer vocab/pre-tokenization sweep, author-ID hyper-parameter sweep):

```bash
python experiments.py # writes outputs/experiment_results.txt
```

---

## Final model configuration

Chosen from the sweep in `experiments.py` (unigram, bigram and trigram all reach
~100% held-out accuracy; we commit to bigram because it genuinely uses n-gram
context):

- N-gram order **n = 2** (bigram)
- Add-k smoothing **k = 0.1**
- BPE vocabulary size **2000**
- Pre-tokenization **whitespace**

These live as the `FINAL_*` constants at the top of `classify_testset.py` and
`evaluate.py` — change them there to try other configurations.
