# Ideally, we make 2 folders, augmented_test_data and augmented_train_data with the new data on top of the existing sheets.
# Also, note that all the rows added via augmentation should have a new column called "augmented" with a value of 1, while the original data should have a value of 0.
#
# Dual Translation - Use this to 2x everything
#
# Counterfactual / Directional Word Swap on 20% of hawkish and dovish data
#
# Synonym Replacement on 75% of the hawkish and dovish data
#
# Masking on 30% of all data
#
# Before any augmentation, training rows are deduplicated: exact duplicate
# sentences within a train file are dropped, as are train sentences that also
# appear in the corresponding test file (train/test leakage). Matching is done
# on a normalized form (lowercased, whitespace collapsed).

import argparse
import json
import random
import re
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
TEST_DIR = SCRIPT_DIR.parent / "test_data"
TRAIN_DIR = SCRIPT_DIR.parent / "training_data"
OUT_TEST_DIR = SCRIPT_DIR / "augmented_test_data"
OUT_TRAIN_DIR = SCRIPT_DIR / "augmented_train_data"

# Label scheme of the lab-manual FOMC dataset
LABEL_DOVISH = 0
LABEL_HAWKISH = 1
LABEL_NEUTRAL = 2

DEFAULT_SEED = 9444
DEFAULT_DEVICE = "cpu"  # translation models; use "cuda" if a GPU is available

COUNTERFACTUAL_FRAC = 0.20  # of hawkish + dovish rows
SYNONYM_FRAC = 0.75         # of hawkish + dovish rows
MASK_FRAC = 0.30            # of all rows
MASK_WORD_RATE = 0.15       # tokens masked within a selected sentence
SYNONYM_WORD_RATE = 0.15    # tokens replaced within a selected sentence
MASK_TOKEN = "[MASK]"

# ---------------------------------------------------------------------------
# Counterfactual / Directional Word Swap
# ---------------------------------------------------------------------------
# Flipping the direction of the key indicator words inverts the stance of the
# sentence, so the swapped copy gets the opposite label (dovish <-> hawkish).
_DIRECTION_PAIRS = [
    ("rising", "falling"), ("rise", "fall"), ("rises", "falls"), ("rose", "fell"), ("risen", "fallen"),
    ("increasing", "decreasing"), ("increase", "decrease"), ("increases", "decreases"), ("increased", "decreased"),
    ("building", "easing"), ("elevated", "moderating"),
    ("higher", "lower"), ("high", "low"),
    ("tight", "soft"), ("tighter", "softer"), ("tightness", "slack"),
    ("tightening", "easing"), ("tighten", "ease"), ("tightened", "eased"),
    ("strong", "weak"), ("stronger", "weaker"), ("strongest", "weakest"), ("strength", "weakness"),
    ("robust", "slack"),
    ("restrictive", "accommodative"), ("contractionary", "expansionary"),
    ("hawkish", "dovish"),
    ("raise", "cut"), ("raised", "cut"), ("raising", "cutting"), ("hike", "cut"), ("hikes", "cuts"),
    ("upside", "downside"), ("upward", "downward"),
    ("accelerating", "decelerating"), ("accelerated", "decelerated"), ("accelerate", "decelerate"),
    ("overheating", "cooling"),
    ("inflationary", "disinflationary"),
]

DIRECTION_MAP = {}
for a, b in _DIRECTION_PAIRS:
    DIRECTION_MAP.setdefault(a, b)
    DIRECTION_MAP.setdefault(b, a)

_DIRECTION_RE = re.compile(
    r"\b(?:" + "|".join(sorted(DIRECTION_MAP, key=len, reverse=True)) + r")\b",
    flags=re.IGNORECASE,
)

_WORD_RE = re.compile(r"[A-Za-z]+")

_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "of", "at", "by", "for", "with",
    "about", "to", "from", "in", "on", "as", "is", "are", "was", "were", "be",
    "been", "being", "that", "this", "these", "those", "it", "its", "we", "our",
    "i", "he", "she", "they", "their", "his", "her", "you", "your", "not", "no",
    "will", "would", "should", "could", "may", "might", "can", "shall", "do",
    "does", "did", "have", "has", "had", "which", "who", "whom", "what", "when",
    "where", "how", "than", "then", "there", "here", "so", "such", "some", "any",
    "all", "both", "each", "more", "most", "other", "into", "over", "under",
    "very", "also", "just", "only", "own", "same", "too",
}

# Domain terms that carry the stance signal - never touched by synonym
# replacement so the label stays valid.
_PROTECTED_WORDS = set(DIRECTION_MAP) | {
    "inflation", "deflation", "disinflation", "unemployment", "employment",
    "labor", "rates", "rate", "policy", "monetary", "fomc", "committee",
    "federal", "reserve", "fed", "funds", "growth", "gdp", "prices", "price",
    "wage", "wages", "recession", "expansion",
}


def _match_case(template: str, word: str) -> str:
    if template.isupper():
        return word.upper()
    if template[:1].isupper():
        return word.capitalize()
    return word


def directional_swap(sentence: str) -> tuple[str, int]:
    """Flip every directional keyword. Returns (new_sentence, n_swaps)."""
    n = 0

    def repl(m: re.Match) -> str:
        nonlocal n
        n += 1
        return _match_case(m.group(0), DIRECTION_MAP[m.group(0).lower()])

    return _DIRECTION_RE.sub(repl, sentence), n


# ---------------------------------------------------------------------------
# Synonym Replacement (WordNet)
# ---------------------------------------------------------------------------
_wordnet = None


def _get_wordnet():
    global _wordnet
    if _wordnet is None:
        import nltk
        from nltk.corpus import wordnet

        for corpus in ("wordnet", "omw-1.4"):
            try:
                nltk.data.find(f"corpora/{corpus}")
            except LookupError:
                nltk.download(corpus, quiet=True)
        wordnet.ensure_loaded()
        _wordnet = wordnet
    return _wordnet


def _synonyms(word: str) -> list[str]:
    wordnet = _get_wordnet()
    out = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            name = lemma.name().replace("_", " ").lower()
            if name != word and name.isalpha():
                out.add(name)
    return sorted(out)


def synonym_replace(sentence: str, rng: random.Random) -> tuple[str, int]:
    """Replace ~SYNONYM_WORD_RATE of eligible words with WordNet synonyms."""
    tokens = list(_WORD_RE.finditer(sentence))
    candidates = [
        m for m in tokens
        if len(m.group(0)) > 2
        and m.group(0).lower() not in _STOPWORDS
        and m.group(0).lower() not in _PROTECTED_WORDS
    ]
    if not candidates:
        return sentence, 0
    n_target = max(1, round(len(candidates) * SYNONYM_WORD_RATE))
    rng.shuffle(candidates)

    replacements = {}  # start position -> (match, new_word)
    for m in candidates:
        if len(replacements) >= n_target:
            break
        syns = _synonyms(m.group(0).lower())
        if syns:
            replacements[m.start()] = (m, _match_case(m.group(0), rng.choice(syns)))

    if not replacements:
        return sentence, 0

    out, prev = [], 0
    for start in sorted(replacements):
        m, new_word = replacements[start]
        out.append(sentence[prev:m.start()])
        out.append(new_word)
        prev = m.end()
    out.append(sentence[prev:])
    return "".join(out), len(replacements)


# ---------------------------------------------------------------------------
# Masking
# ---------------------------------------------------------------------------
def mask_sentence(sentence: str, rng: random.Random) -> tuple[str, int]:
    """Replace ~MASK_WORD_RATE of words with the [MASK] token."""
    tokens = list(_WORD_RE.finditer(sentence))
    if not tokens:
        return sentence, 0
    n_target = max(1, round(len(tokens) * MASK_WORD_RATE))
    chosen = sorted(rng.sample(range(len(tokens)), min(n_target, len(tokens))))

    out, prev = [], 0
    for i in chosen:
        m = tokens[i]
        out.append(sentence[prev:m.start()])
        out.append(MASK_TOKEN)
        prev = m.end()
    out.append(sentence[prev:])
    return "".join(out), len(chosen)


# ---------------------------------------------------------------------------
# Dual (back-) translation: EN -> DE -> EN
# ---------------------------------------------------------------------------
_translators = None

# Back-translations are deterministic (greedy decoding), so they are cached on
# disk - reruns and interrupted runs only translate sentences not seen before.
_BT_CACHE_PATH = SCRIPT_DIR / "bt_cache.json"
_bt_cache: dict[str, str] | None = None


def _get_bt_cache() -> dict[str, str]:
    global _bt_cache
    if _bt_cache is None:
        if _BT_CACHE_PATH.exists():
            _bt_cache = json.loads(_BT_CACHE_PATH.read_text(encoding="utf-8"))
        else:
            _bt_cache = {}
    return _bt_cache


def _get_translators(device: str):
    global _translators
    if _translators is None:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        print("  loading translation models (Helsinki-NLP opus-mt en<->de)...")
        pairs = []
        for name in ("Helsinki-NLP/opus-mt-en-de", "Helsinki-NLP/opus-mt-de-en"):
            tok = AutoTokenizer.from_pretrained(name)
            model = AutoModelForSeq2SeqLM.from_pretrained(name).to(device).eval()
            pairs.append((tok, model))
        _translators = tuple(pairs)
    return _translators


def _translate(texts: list[str], tok, model, batch_size: int, desc: str) -> list[str]:
    import torch

    out = []
    for i in range(0, len(texts), batch_size):
        batch = tok(texts[i:i + batch_size], return_tensors="pt",
                    padding=True, truncation=True, max_length=512).to(model.device)
        with torch.no_grad():
            generated = model.generate(**batch, max_length=512, num_beams=1)
        out.extend(tok.batch_decode(generated, skip_special_tokens=True))
        print(f"    {desc}: {len(out)}/{len(texts)}", flush=True)
    return out


def back_translate(sentences: list[str], device: str, batch_size: int = 32) -> list[str]:
    """EN -> DE -> EN for each sentence, with caching across files and runs.

    The cache is flushed to disk after every chunk, so an interrupted run
    loses at most one chunk of work.
    """
    cache = _get_bt_cache()
    todo = [s for s in dict.fromkeys(sentences) if s not in cache]
    if todo:
        (en_de_tok, en_de), (de_en_tok, de_en) = _get_translators(device)
        print(f"  back-translating {len(todo)} new unique sentences...")
        chunk_size = batch_size * 4
        for i in range(0, len(todo), chunk_size):
            chunk = todo[i:i + chunk_size]
            desc = f"[{i + len(chunk)}/{len(todo)}]"
            german = _translate(chunk, en_de_tok, en_de, batch_size, f"en->de {desc}")
            english = _translate(german, de_en_tok, de_en, batch_size, f"de->en {desc}")
            cache.update(zip(chunk, english))
            _BT_CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    return [cache[s] for s in sentences]


# ---------------------------------------------------------------------------
# Per-file augmentation
# ---------------------------------------------------------------------------
def _sample(df: pd.DataFrame, frac: float, rng: random.Random) -> pd.DataFrame:
    n = round(len(df) * frac)
    if n == 0:
        return df.iloc[0:0]
    return df.sample(n=n, random_state=rng.randrange(2**32))


def _norm_sentence(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip().lower()


def dedupe_train(df: pd.DataFrame, test_path: Path) -> pd.DataFrame:
    """Drop intra-train duplicate sentences and train/test overlap."""
    norm = df["sentence"].map(_norm_sentence)
    df = df[~norm.duplicated(keep="first")]
    n_intra = len(norm) - len(df)

    test_sentences = set(pd.read_excel(test_path)["sentence"].map(_norm_sentence))
    keep = ~df["sentence"].map(_norm_sentence).isin(test_sentences)
    n_leak = (~keep).sum()

    print(f"  dedup: dropped {n_intra} intra-train duplicates, {n_leak} train/test overlaps")
    return df[keep].reset_index(drop=True)


def augment_train_file(df: pd.DataFrame, rng: random.Random, device: str) -> pd.DataFrame:
    df = df.copy()
    df["augmented"] = 0
    df["aug_method"] = "original"

    # Dual Translation - doubles the pool.
    translated = df.copy()
    translated["sentence"] = back_translate(df["sentence"].tolist(), device)
    translated["augmented"] = 1
    translated["aug_method"] = "back_translation"

    new_rows = []
    minority = df[df["label"].isin([LABEL_DOVISH, LABEL_HAWKISH])]

    # Counterfactual / Directional Word Swap on 20% of hawkish and dovish data.
    # The swap inverts the stance, so the label flips dovish <-> hawkish.
    eligible = minority[minority["sentence"].str.contains(_DIRECTION_RE, regex=True)]
    for label in (LABEL_DOVISH, LABEL_HAWKISH):
        pool = eligible[eligible["label"] == label]
        n_class = (minority["label"] == label).sum()
        take = min(round(n_class * COUNTERFACTUAL_FRAC), len(pool))
        for _, row in pool.sample(n=take, random_state=rng.randrange(2**32)).iterrows():
            swapped, n = directional_swap(row["sentence"])
            if n == 0:
                continue
            new = row.copy()
            new["sentence"] = swapped
            new["label"] = LABEL_HAWKISH if label == LABEL_DOVISH else LABEL_DOVISH
            new["augmented"] = 1
            new["aug_method"] = "directional_swap"
            new_rows.append(new)

    # Synonym Replacement on 75% of the hawkish and dovish data.
    for _, row in _sample(minority, SYNONYM_FRAC, rng).iterrows():
        replaced, n = synonym_replace(row["sentence"], rng)
        if n == 0:
            continue
        new = row.copy()
        new["sentence"] = replaced
        new["augmented"] = 1
        new["aug_method"] = "synonym_replacement"
        new_rows.append(new)

    # Masking on 30% of all data.
    for _, row in _sample(df, MASK_FRAC, rng).iterrows():
        masked, n = mask_sentence(row["sentence"], rng)
        if n == 0:
            continue
        new = row.copy()
        new["sentence"] = masked
        new["augmented"] = 1
        new["aug_method"] = "masking"
        new_rows.append(new)

    return pd.concat([df, translated, pd.DataFrame(new_rows)], ignore_index=True)


def augment_test_file(df: pd.DataFrame, device: str) -> pd.DataFrame:
    df = df.copy()
    df["augmented"] = 0
    df["aug_method"] = "original"

    # Dual Translation - doubles the test data pool.
    translated = df.copy()
    translated["sentence"] = back_translate(df["sentence"].tolist(), device)
    translated["augmented"] = 1
    translated["aug_method"] = "back_translation"

    return pd.concat([df, translated], ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description="Augment the FOMC train/test xlsx datasets.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="RNG seed for reproducibility")
    parser.add_argument("--device", default=DEFAULT_DEVICE, help="device for translation models, e.g. cpu / cuda / cuda:0")
    parser.add_argument("--skip-test", action="store_true", help="skip test-data back-translation")
    parser.add_argument("--skip-train", action="store_true", help="skip training-data augmentation")
    args = parser.parse_args()

    OUT_TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    OUT_TEST_DIR.mkdir(parents=True, exist_ok=True)

    if not args.skip_train:
        for path in sorted(TRAIN_DIR.glob("*.xlsx")):
            print(f"[train] {path.name}")
            rng = random.Random(args.seed)

            df = pd.read_excel(path)
            df = dedupe_train(df, TEST_DIR / path.name.replace("-train-", "-test-"))
            out = augment_train_file(df, rng, args.device)

            out.to_excel(OUT_TRAIN_DIR / path.name, index=False)
            print(f"  {len(df)} -> {len(out)} rows "
                  f"({out['aug_method'].value_counts().drop('original').to_dict()})")

    if not args.skip_test:
        for path in sorted(TEST_DIR.glob("*.xlsx")):
            print(f"[test] {path.name}")

            df = pd.read_excel(path)
            out = augment_test_file(df, args.device)

            out.to_excel(OUT_TEST_DIR / path.name, index=False)
            print(f"  {len(df)} -> {len(out)} rows")

    print("done.")


if __name__ == "__main__":
    main()
