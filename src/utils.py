"""utils for corpus splitting and building held-out passages for evaluation.
These utils help us to
  * measure perplexity on unseen text
  * build a labelled pseudo test-set of short passages to measure author-ID
    accuracy
"""


def split_lines_train_test(text, test_frac=0.1, seed=13):
    """Split a corpus into train/test by lines.

    We take every Nth line for the test set (N comes from test_frac), so both
    parts come from all through the book instead of one chunk each.
    """
    # 1. Get the lines, skipping the blank ones.
    all_lines = text.splitlines()
    lines = []
    for line in all_lines:
        if line.strip() != "":
            lines.append(line)
    # 2. Work out how often to take a test line.
    # test_frac=0.1 means 1 line in every 10, so step = 10.
    step = int(round(1 / test_frac))
    if step < 2:
        # Guard: step of 1 would put every line in the test set.
        step = 2
    # 3. `seed` just decides which lines in each group of `step` are test ones.
    offset = seed % step
    # 4. Walk through the lines and put each one in a bucket.
    train_lines = []
    test_lines = []
    for i in range(len(lines)):
        line = lines[i]
        if i % step == offset:
            test_lines.append(line)
        else:
            train_lines.append(line)
    # 5. Glue each bucket back into one big string.
    train_text = "\n".join(train_lines)
    test_text = "\n".join(test_lines)
    return train_text, test_text

def make_passages(text, words_per_passage=40, max_passages=60):
    """Cut the text into short chunks of the same number of words.

    The real test set is made of short passages, so we make ours the same
    shape. That way our accuracy score means something.
    """
    # Split on whitespace. Punctuation stays stuck to the words, which is fine
    # because we only need to count words here.
    words = text.split()
    passages = []
    start = 0
    while start < len(words) - words_per_passage:
        # Take exactly `words_per_passage` words, starting at `start`.
        chunk = words[start:start + words_per_passage]
        passages.append(" ".join(chunk))
        # Stop early if we already have enough passages.
        if len(passages) >= max_passages:
            break
        # Move along to the next chunk (no overlap).
        start = start + words_per_passage

    return passages
