"""Build fed_corpus.txt for dapt.py (step 2) and self_train.py (step 5).

Source: the dataset authors' raw sentence dumps (gtfintechlab/fomc-hawkish-dovish,
data/filtered_data) - every FOMC meeting minute, press conference and speech
sentence, 1996-2022. This is the same corpus the labeled dataset was sampled
from, which makes it the ideal DAPT text: identical domain by construction.

Filters applied, in order:
1. drop sentences with no letters or fewer than --min-words words
2. dedupe on normalized form (lowercase, whitespace collapsed)
3. drop anything matching a test sentence in ANY test file (all variants,
   seeds, and augmented test sets) - the project's hard leakage rule

  python build_corpus.py --source <path-to-clone>/data/filtered_data
"""

import argparse
import csv
from pathlib import Path

import common

csv.field_size_limit(10_000_000)

# speech/ has all|select|non-select subsets; "all" is their union
SOURCE_DIRS = ["meeting_minutes", "press_conference", "speech/all"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True,
                        help="path to the clone's data/filtered_data directory")
    parser.add_argument("--output", type=Path, default=common.TOM_DIR / "fed_corpus.txt")
    parser.add_argument("--min-words", type=int, default=5)
    args = parser.parse_args()

    print("collecting test sentences to exclude...")
    test_norm = common.all_test_sentences()
    print(f"  {len(test_norm)} unique test sentences")

    seen: set[str] = set()
    kept: list[str] = []
    n_raw = n_short = n_test = 0
    for sub in SOURCE_DIRS:
        files = sorted((args.source / sub).glob("*.csv"))
        sub_kept = 0
        for path in files:
            with open(path, encoding="utf-8", errors="replace", newline="") as f:
                for row in csv.DictReader(f):
                    s = (row.get("sentence") or "").strip()
                    n_raw += 1
                    if len(s.split()) < args.min_words or not any(c.isalpha() for c in s):
                        n_short += 1
                        continue
                    key = common.norm_sentence(s)
                    if key in seen:
                        continue
                    if key in test_norm:
                        n_test += 1
                        continue
                    seen.add(key)
                    kept.append(" ".join(s.split()))
                    sub_kept += 1
        print(f"  {sub}: {len(files)} files, {sub_kept} sentences kept")

    args.output.write_text("\n".join(kept) + "\n", encoding="utf-8")
    words = sum(len(s.split()) for s in kept)
    print(f"\nraw rows: {n_raw} | dropped short/garbage: {n_short} | "
          f"dropped test matches: {n_test} | duplicates: {n_raw - n_short - n_test - len(kept)}")
    print(f"corpus: {len(kept):,} sentences, ~{words:,} words -> {args.output}")

    # hard verification of the leakage rule
    assert not any(common.norm_sentence(s) in test_norm for s in kept), "test sentence survived filtering!"
    print("verified: zero test sentences in corpus")


if __name__ == "__main__":
    main()
