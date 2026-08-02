"""Shared RoBERTa experiment runner for focused FOMC ablations."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import (
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup,
)

MODELS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = MODELS_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from fomc_dataset_analysis import load_seed as load_all_seed_records

SEEDS = ["5768", "78516", "944601"]
LABEL_NAMES = ["Dovish", "Hawkish", "Neutral"]
HP_GRID = [
    {"learning_rate": 1e-5, "batch_size": 8},
    {"learning_rate": 2e-5, "batch_size": 8},
    {"learning_rate": 5e-5, "batch_size": 8},
    {"learning_rate": 1e-5, "batch_size": 16},
    {"learning_rate": 2e-5, "batch_size": 16},
    {"learning_rate": 5e-5, "batch_size": 16},
]


@dataclass(frozen=True)
class VariantConfig:
    name: str
    use_dapt: bool = False
    use_average_weighted_hp_search: bool = False
    use_class_balancing: bool = False
    label_smoothing: float = 0.0


@dataclass(frozen=True)
class TrainConfig:
    checkpoint: str = "roberta-base"
    max_length: int = 256
    max_epochs: int = 10
    patience: int = 2
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    dropout: float = 0.1
    dapt_epochs: int = 3
    dapt_learning_rate: float = 5e-5
    mlm_probability: float = 0.15


class ClassificationDataset(Dataset):
    def __init__(self, records: list[dict], tokenizer, max_length: int):
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        encoded = self.tokenizer(
            normalise(record["sentence"]),
            max_length=self.max_length,
            truncation=True,
            padding=False,
        )
        return {**encoded, "label": record["label"]}


class TextDataset(Dataset):
    def __init__(self, texts: list[str], tokenizer, max_length: int):
        self.examples = tokenizer(
            [normalise(text) for text in texts],
            max_length=max_length,
            truncation=True,
            padding=False,
        )["input_ids"]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict:
        return {"input_ids": self.examples[index]}


def normalise(text: str) -> str:
    return " ".join(text.strip().split())


def load_seed(seed: str) -> tuple[list[dict], list[dict]]:
    records = load_all_seed_records(seed)
    return (
        [record for record in records if record["split"] == "train"],
        [record for record in records if record["split"] == "test"],
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def split_train_validation(records: list[dict], seed: int) -> tuple[list[dict], list[dict]]:
    train_indices, validation_indices = train_test_split(
        range(len(records)),
        test_size=0.2,
        random_state=seed,
        stratify=[record["label"] for record in records],
    )
    return (
        [records[index] for index in train_indices],
        [records[index] for index in validation_indices],
    )


def make_collator(tokenizer):
    def collate(batch: list[dict]) -> dict[str, torch.Tensor]:
        labels = torch.tensor([item.pop("label") for item in batch], dtype=torch.long)
        padded = tokenizer.pad(batch, padding=True, return_tensors="pt")
        padded["labels"] = labels
        return padded

    return collate


def compute_class_weights(records: list[dict], device: torch.device) -> torch.Tensor:
    counts = Counter(record["label"] for record in records)
    total = len(records)
    weights = [total / (len(LABEL_NAMES) * counts[label]) for label in range(len(LABEL_NAMES))]
    return torch.tensor(weights, dtype=torch.float32, device=device)


def classification_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    class_weights: torch.Tensor | None,
    label_smoothing: float,
) -> torch.Tensor:
    return F.cross_entropy(
        logits,
        labels,
        weight=class_weights,
        label_smoothing=label_smoothing,
    )


def run_dapt(
    records: list[dict],
    output_dir: Path,
    seed: int,
    train_config: TrainConfig,
    device: torch.device,
) -> Path:
    marker = output_dir / "config.json"
    if marker.exists():
        print(f"Reusing cached DAPT checkpoint: {output_dir}")
        return output_dir

    set_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(train_config.checkpoint)
    dataset = TextDataset([record["sentence"] for record in records], tokenizer, train_config.max_length)
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=train_config.mlm_probability,
    )
    loader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collator)
    model = AutoModelForMaskedLM.from_pretrained(train_config.checkpoint).to(device)
    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.dapt_learning_rate,
        weight_decay=train_config.weight_decay,
    )
    total_steps = max(len(loader) * train_config.dapt_epochs, 1)
    scheduler = get_linear_schedule_with_warmup(
        optimiser,
        num_warmup_steps=int(total_steps * train_config.warmup_ratio),
        num_training_steps=total_steps,
    )

    model.train()
    with tqdm(total=total_steps, desc=f"DAPT seed={seed}", unit="batch", dynamic_ncols=True) as progress:
        for epoch in range(1, train_config.dapt_epochs + 1):
            total_loss = 0.0
            for batch in loader:
                batch = {key: value.to(device) for key, value in batch.items()}
                optimiser.zero_grad()
                loss = model(**batch).loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimiser.step()
                scheduler.step()
                total_loss += loss.item()
                progress.update()
                progress.set_postfix(epoch=f"{epoch}/{train_config.dapt_epochs}", mlm_loss=f"{loss.item():.4f}")
            progress.write(f"DAPT epoch {epoch}: MLM loss={total_loss / max(len(loader), 1):.4f}")

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    return output_dir


def evaluate(
    model,
    loader: DataLoader,
    device: torch.device,
    class_weights: torch.Tensor | None,
    label_smoothing: float,
) -> dict:
    model.eval()
    losses: list[float] = []
    predictions: list[int] = []
    labels: list[int] = []
    probabilities: list[list[float]] = []
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            output = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            loss = classification_loss(output.logits, batch["labels"], class_weights, label_smoothing)
            probs = torch.softmax(output.logits, dim=-1)
            losses.append(loss.item())
            predictions.extend(probs.argmax(dim=-1).cpu().tolist())
            labels.extend(batch["labels"].cpu().tolist())
            probabilities.extend(probs.cpu().tolist())
    return {
        "loss": float(np.mean(losses)),
        "weighted_f1": f1_score(labels, predictions, average="weighted", zero_division=0),
        "macro_f1": f1_score(labels, predictions, average="macro", zero_division=0),
        "accuracy": accuracy_score(labels, predictions),
        "predictions": predictions,
        "labels": labels,
        "probabilities": probabilities,
    }


def train_classifier(
    seed: str,
    variant: VariantConfig,
    train_config: TrainConfig,
    learning_rate: float,
    batch_size: int,
    output_dir: Path,
    device: torch.device,
    save_outputs: bool = True,
) -> dict:
    integer_seed = int(seed)
    set_seed(integer_seed)
    all_train_records, test_records = load_seed(seed)
    train_records, validation_records = split_train_validation(all_train_records, integer_seed)

    checkpoint: str | Path = train_config.checkpoint
    if variant.use_dapt:
        checkpoint = run_dapt(
            train_records,
            PROJECT_DIR / "results" / variant.name / "_dapt" / f"seed_{seed}",
            integer_seed,
            train_config,
            device,
        )

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    collator = make_collator(tokenizer)
    generator = torch.Generator().manual_seed(integer_seed)
    train_loader = DataLoader(
        ClassificationDataset(train_records, tokenizer, train_config.max_length),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=collator,
    )
    validation_loader = DataLoader(
        ClassificationDataset(validation_records, tokenizer, train_config.max_length),
        batch_size=batch_size,
        collate_fn=collator,
    )
    test_loader = DataLoader(
        ClassificationDataset(test_records, tokenizer, train_config.max_length),
        batch_size=batch_size,
        collate_fn=collator,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint,
        num_labels=len(LABEL_NAMES),
        hidden_dropout_prob=train_config.dropout,
        attention_probs_dropout_prob=train_config.dropout,
        ignore_mismatched_sizes=True,
    ).to(device)
    class_weights = compute_class_weights(train_records, device) if variant.use_class_balancing else None
    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=train_config.weight_decay,
    )
    total_steps = max(len(train_loader) * train_config.max_epochs, 1)
    scheduler = get_linear_schedule_with_warmup(
        optimiser,
        num_warmup_steps=int(total_steps * train_config.warmup_ratio),
        num_training_steps=total_steps,
    )

    best_score = -1.0
    best_epoch = 0
    best_state = None
    patience = 0
    history: list[dict] = []
    started = time.time()
    with tqdm(
        total=total_steps,
        desc=f"{variant.name} seed={seed}",
        unit="batch",
        dynamic_ncols=True,
    ) as progress:
        for epoch in range(1, train_config.max_epochs + 1):
            model.train()
            train_losses: list[float] = []
            for batch in train_loader:
                batch = {key: value.to(device) for key, value in batch.items()}
                optimiser.zero_grad()
                logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
                loss = classification_loss(logits, batch["labels"], class_weights, variant.label_smoothing)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimiser.step()
                scheduler.step()
                train_losses.append(loss.item())
                progress.update()
                progress.set_postfix(epoch=f"{epoch}/{train_config.max_epochs}", loss=f"{loss.item():.4f}")

            validation = evaluate(model, validation_loader, device, class_weights, variant.label_smoothing)
            history.append({
                "epoch": epoch,
                "train_loss": float(np.mean(train_losses)),
                "validation_loss": validation["loss"],
                "validation_weighted_f1": validation["weighted_f1"],
            })
            progress.set_postfix(
                epoch=f"{epoch}/{train_config.max_epochs}",
                loss=f"{history[-1]['train_loss']:.4f}",
                val_f1=f"{validation['weighted_f1']:.4f}",
            )
            progress.write(
                f"{variant.name} seed={seed} epoch={epoch} "
                f"train_loss={history[-1]['train_loss']:.4f} "
                f"val_weighted_f1={validation['weighted_f1']:.4f}"
            )
            if validation["weighted_f1"] > best_score:
                best_score = validation["weighted_f1"]
                best_epoch = epoch
                best_state = {key: value.cpu().clone() for key, value in model.state_dict().items()}
                patience = 0
            else:
                patience += 1
                if patience >= train_config.patience:
                    progress.set_description(f"{variant.name} seed={seed} early stop")
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
    metrics = {
        "seed": seed,
        "variant": variant.name,
        "validation_weighted_f1": best_score,
        "best_epoch": best_epoch,
        "training_seconds": time.time() - started,
    }
    if not save_outputs:
        return metrics

    test = evaluate(model, test_loader, device, class_weights, variant.label_smoothing)
    metrics.update({
        "test_weighted_f1": test["weighted_f1"],
        "test_macro_f1": test["macro_f1"],
        "test_accuracy": test["accuracy"],
    })
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir / "model")
    tokenizer.save_pretrained(output_dir / "model")
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    with (output_dir / "config.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "variant": asdict(variant),
                "training": asdict(train_config),
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "class_weights": class_weights.cpu().tolist() if class_weights is not None else None,
            },
            file,
            indent=2,
        )
    with (output_dir / "training_history.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    with (output_dir / "classification_report.json").open("w", encoding="utf-8") as file:
        json.dump(
            classification_report(
                test["labels"],
                test["predictions"],
                target_names=LABEL_NAMES,
                output_dict=True,
                zero_division=0,
            ),
            file,
            indent=2,
        )
    return metrics


def average_weighted_hyperparameter_search(
    variant: VariantConfig,
    train_config: TrainConfig,
    results_dir: Path,
    device: torch.device,
) -> tuple[float, int]:
    trials: list[dict] = []
    for hyperparameters in tqdm(HP_GRID, desc=f"{variant.name} HP trials", unit="trial", dynamic_ncols=True):
        scores = []
        for seed in SEEDS:
            metrics = train_classifier(
                seed,
                variant,
                train_config,
                hyperparameters["learning_rate"],
                hyperparameters["batch_size"],
                results_dir / "_hp_search" / f"lr_{hyperparameters['learning_rate']}_bs_{hyperparameters['batch_size']}" / f"seed_{seed}",
                device,
                save_outputs=False,
            )
            scores.append(metrics["validation_weighted_f1"])
        trial = {
            **hyperparameters,
            "per_seed_validation_weighted_f1": dict(zip(SEEDS, scores)),
            "mean_validation_weighted_f1": float(np.mean(scores)),
            "std_validation_weighted_f1": float(np.std(scores)),
        }
        trials.append(trial)
        print(f"HP trial: {trial}")

    results_dir.mkdir(parents=True, exist_ok=True)
    with (results_dir / "hyperparameter_search.json").open("w", encoding="utf-8") as file:
        json.dump(trials, file, indent=2)
    best = max(trials, key=lambda trial: trial["mean_validation_weighted_f1"])
    return best["learning_rate"], best["batch_size"]


def run_variant(variant: VariantConfig) -> None:
    parser = argparse.ArgumentParser(description=f"Run {variant.name} FOMC experiment")
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--skip-hp-search", action="store_true")
    parser.add_argument("--seeds", nargs="+", choices=SEEDS, default=SEEDS)
    args = parser.parse_args()

    train_config = TrainConfig()
    device = get_device()
    results_dir = PROJECT_DIR / "results" / variant.name
    learning_rate, batch_size = args.learning_rate, args.batch_size
    if variant.use_average_weighted_hp_search and not args.skip_hp_search:
        learning_rate, batch_size = average_weighted_hyperparameter_search(
            variant,
            train_config,
            results_dir,
            device,
        )

    seed_metrics = []
    for seed in args.seeds:
        seed_metrics.append(
            train_classifier(
                seed,
                variant,
                train_config,
                learning_rate,
                batch_size,
                results_dir / f"seed_{seed}",
                device,
            )
        )
    aggregate = {
        "variant": variant.name,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "seeds": args.seeds,
        "test_weighted_f1_mean": float(np.mean([item["test_weighted_f1"] for item in seed_metrics])),
        "test_weighted_f1_std": float(np.std([item["test_weighted_f1"] for item in seed_metrics])),
        "test_macro_f1_mean": float(np.mean([item["test_macro_f1"] for item in seed_metrics])),
        "test_accuracy_mean": float(np.mean([item["test_accuracy"] for item in seed_metrics])),
        "per_seed": seed_metrics,
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    with (results_dir / "aggregate_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(aggregate, file, indent=2)
    print(json.dumps(aggregate, indent=2))
