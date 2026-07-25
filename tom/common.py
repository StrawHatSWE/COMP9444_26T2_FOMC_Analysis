"""Shared infrastructure for the tom pipeline (RoBERTa FOMC stance model).
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
)

TOM_DIR = Path(__file__).resolve().parent
DATASET_ROOT = TOM_DIR.parent / "dataset" / "test-and-training"
RUNS_DIR = TOM_DIR / "runs"

ID_TO_LABEL = {0: "dovish", 1: "hawkish", 2: "neutral"}
LABEL_TO_ID = {v: k for k, v in ID_TO_LABEL.items()}

SEARCH_SEED = 5768                      # config search happens ONLY on this seed
FINAL_SEEDS = [5768, 78516, 944601]
VAL_FRACTION = 0.2
MAX_LENGTH = 128


def data_dirs(augmented: bool) -> tuple[Path, Path]:
    if augmented:
        base = DATASET_ROOT / "augmented_data"
        return base / "augmented_train_data", base / "augmented_test_data"
    return DATASET_ROOT / "training_data", DATASET_ROOT / "test_data"


def norm_sentence(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip().lower()


def _xlsx_files(directory: Path) -> list[Path]:
    """All real workbooks in a directory, skipping Excel ~$ lock files."""
    return sorted(p for p in directory.glob("*.xlsx") if not p.name.startswith("~$"))


def all_test_sentences() -> set[str]:
    """Normalized sentences from EVERY test file (all variants, seeds, and the
    augmented test sets). Anything in this set must never appear in any
    training-side corpus."""
    out = set()
    for d in (DATASET_ROOT / "test_data", DATASET_ROOT / "augmented_data" / "augmented_test_data"):
        for path in _xlsx_files(d):
            out |= set(pd.read_excel(path)["sentence"].map(norm_sentence))
    return out


_TEST_DIR_NAME = {"training_data": "test_data", "augmented_train_data": "augmented_test_data"}


def load_split(train_path: Path, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Family-safe train/val split of one training file. This is cuz this method

    - dedups rows and drops train/test overlap (vs the matching test file)
    - validation comes from original (augmented == 0) rows only
    - augmented rows whose source (index, year) landed in validation are dropped
    """
    df = pd.read_excel(train_path)

    n_before = len(df)
    norm = df["sentence"].map(norm_sentence)
    df = df[~norm.duplicated(keep="first")]
    test_path = (train_path.parent.parent / _TEST_DIR_NAME[train_path.parent.name]
                 / train_path.name.replace("-train-", "-test-"))
    test_norm = set(pd.read_excel(test_path)["sentence"].map(norm_sentence))
    df = df[~df["sentence"].map(norm_sentence).isin(test_norm)]
    if len(df) != n_before:
        print(f"[seed {seed}] dedup: dropped {n_before - len(df)} duplicate/leaked rows")

    if "augmented" in df.columns:
        originals = df[df["augmented"] == 0]
        augmented = df[df["augmented"] == 1]
    else:
        originals, augmented = df, df.iloc[0:0]

    train_orig, val_df = train_test_split(
        originals, test_size=VAL_FRACTION, random_state=seed, stratify=originals["label"],
    )

    parts = [train_orig]
    if len(augmented):
        val_keys = set(zip(val_df["index"], val_df["year"]))
        keep = [key not in val_keys for key in zip(augmented["index"], augmented["year"])]
        parts.append(augmented[keep])

    train_df = pd.concat(parts, ignore_index=True)
    return train_df, val_df.reset_index(drop=True)


def unlabeled_corpus() -> list[str]:
    """Fallback TAPT pool: training-file sentences filtered against every test
    sentence. Tiny (~257 sentences) - prefer dapt.py --corpus fed_corpus.txt."""
    test_norm = all_test_sentences()
    seen, out = set(), []
    for path in _xlsx_files(data_dirs(False)[0]):
        for s in pd.read_excel(path)["sentence"].astype(str):
            key = norm_sentence(s)
            if key in test_norm or key in seen or not re.search(r"[A-Za-z]", s):
                continue
            seen.add(key)
            out.append(s.strip())
    return out


def load_tokenizer_and_model(model_name_or_path: str, num_labels: int = 3):
    tok = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path, num_labels=num_labels,
        id2label=ID_TO_LABEL, label2id=LABEL_TO_ID,
    )
    return tok, model


def to_dataset(df: pd.DataFrame, tokenizer) -> Dataset:
    ds = Dataset.from_pandas(df[["sentence", "label"]].reset_index(drop=True))
    ds = ds.map(lambda ex: tokenizer(ex["sentence"], truncation=True, max_length=MAX_LENGTH), batched=True)
    return ds.rename_columns({"label": "labels"})


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="weighted", zero_division=0)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "macro_f1": f1_score(labels, predictions, average="macro", zero_division=0),
    }


def class_weights(train_df: pd.DataFrame) -> torch.Tensor:
    counts = train_df["label"].value_counts().sort_index()
    return torch.tensor((len(train_df) / (3 * counts)).values, dtype=torch.float32)


class WeightedTrainer(Trainer):
    """Class-weighted cross-entropy with optional label smoothing."""

    def __init__(self, class_weights: torch.Tensor, label_smoothing: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self._class_weights = class_weights
        self._label_smoothing = label_smoothing

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss_fct = torch.nn.CrossEntropyLoss(
            weight=self._class_weights.to(outputs.logits.device),
            label_smoothing=self._label_smoothing,
        )
        loss = loss_fct(outputs.logits.view(-1, model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


@torch.no_grad()
def predict_proba(model_dir: str | Path, sentences: list[str], batch_size: int = 32,
                  device: str = "cpu") -> np.ndarray:
    """Softmax probabilities from a saved model directory."""
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir)).to(device).eval()
    probs = []
    for i in range(0, len(sentences), batch_size):
        enc = tok(sentences[i:i + batch_size], return_tensors="pt", padding=True,
                  truncation=True, max_length=MAX_LENGTH).to(device)
        logits = model(**enc).logits
        probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
    return np.concatenate(probs)
