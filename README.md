# Language Models and Author Identification

BPE tokenizer (Hugging Face interface) + N-gram language models (bigram/trigram
with add-k smoothing + perplexity) + LM-based author identification for
Tolkien (*The Hobbit*) vs. Conan-Doyle (*The Lost World*).

## Layout

```
data/                     cleaned corpora (Gutenberg boilerplate stripped from Lost World)
  hobbit.txt              Tolkien training text
  lostworld.txt           Conan-Doyle training text
src/
  bpe_tokenizer.py        BPETokenizer: interface to Hugging Face BPE (train/encode->ids/decode)
  ngram_lm.py             NGramLM: counts over vocab ids, add-k smoothing, NLL, perplexity
  author_id.py            AuthorIdentifier: shared tokenizer + one LM per author, classify
  utils.py                train/test splitting and passage helpers
experiments.py            reproduces every table in the report -> outputs/experiment_results.txt
evaluate.py               held-out author-ID eval; also generates our labelled test set
classify_testset.py       Deliverable #3: label a provided test file
testset/                  self-generated held-out test set + answer key
TESTING.md                grader/TA instructions (this info in more detail)
outputs/                  generated results / predictions
```

## Setup

```bash
pip install -r requirements.txt 
```

Python 3.9+; no GPU or internet needed.

## For TAs / graders — running the test set

Full details are in **`TESTING.md`**. Quick version:

**1. Label your test set** (produces the Deliverable-3 labels):

```bash
python classify_testset.py your_testset.txt
```

Writes `outputs/author_id_predictions_labels.txt` (one predicted author per line,
in order) and `outputs/author_id_predictions.csv` (`index, predicted_author,
passage`). Authors are `Tolkien` or `Conan-Doyle`.

**Input format.** By default each **non-empty line is one passage** (the same
format as our generated test set — this is the expected layout, and it just
works). If a passage instead **wraps across several lines** and passages are
separated by blank lines, add `--mode paragraph`:

```bash
python classify_testset.py your_testset.txt --mode paragraph
```

**Optional — score against an answer key** (one true label per line, same order):

```bash
python classify_testset.py your_testset.txt --gold your_answers.txt
```

Prints e.g. `Accuracy vs gold: 47/50 = 0.940`.

**2. Reproduce our held-out evaluation** (no external file needed):

```bash
python evaluate.py  
```

**3. Reproduce the report's tables:**

```bash
python experiments.py
```

## Notes / caveats

- If the test set is one passage per line (same format as ours), the default
  works directly. `--mode paragraph` covers blank-line-separated multi-line
  passages. Those two cover every layout we anticipate.
- Uses only the Hugging Face `tokenizers` BPE (as required) and the Python
  standard library for the modeling; no N-gram/LM library is used.
- The written report (with the AI-generated-code disclosure) is submitted
  separately with the assignment.
