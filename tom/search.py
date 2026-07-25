"""Step 3 - Hyperparameter search, honestly scoped.

Every candidate trains on the seed-5768 training split and is scored on the
seed-5768 validation split ONLY (family-safe split from common.load_split).
Test files are never touched here. The winner (by validation weighted F1,
tie-broken by macro F1) is frozen to selected_config.json for train_final.py.

Candidate list follows the team's one-knob-at-a-time convention, seeded with
the knobs their FinBERT search proved out, plus untested combinations.

  python search.py --model roberta-base
  python search.py --model runs/dapt --augmented
  python search.py --smoke                 # functionality check only
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from transformers import DataCollatorWithPadding, EarlyStoppingCallback, TrainingArguments

import common

CANDIDATES = [
    {"name": "control",        "lr": 2e-5, "batch_size": 32, "label_smoothing": 0.0,  "epochs": 4},
    {"name": "lr3e5",          "lr": 3e-5, "batch_size": 32, "label_smoothing": 0.0,  "epochs": 4},
    {"name": "lr1e5",          "lr": 1e-5, "batch_size": 32, "label_smoothing": 0.0,  "epochs": 4},
    {"name": "smooth05",       "lr": 2e-5, "batch_size": 32, "label_smoothing": 0.05, "epochs": 4},
    {"name": "batch16",        "lr": 2e-5, "batch_size": 16, "label_smoothing": 0.0,  "epochs": 4},
    {"name": "batch8_long",    "lr": 2e-5, "batch_size": 8,  "label_smoothing": 0.0,  "epochs": 8},
    {"name": "lr3e5_smooth",   "lr": 3e-5, "batch_size": 32, "label_smoothing": 0.05, "epochs": 4},
    {"name": "lr3e5_batch8",   "lr": 3e-5, "batch_size": 8,  "label_smoothing": 0.05, "epochs": 8},
    {"name": "lr1e5_batch8",   "lr": 1e-5, "batch_size": 8,  "label_smoothing": 0.05, "epochs": 8},
]

PATIENCE = 2


def run_candidate(cand: dict, args, train_df, val_df) -> dict:
    tok, model = common.load_tokenizer_and_model(args.model)

    training_args = TrainingArguments(
        output_dir=str(common.RUNS_DIR / "search-tmp"),
        num_train_epochs=cand["epochs"],
        per_device_train_batch_size=cand["batch_size"],
        per_device_eval_batch_size=64,
        learning_rate=cand["lr"],
        warmup_ratio=0.1,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        seed=common.SEARCH_SEED,
        report_to=[],
    )
    trainer = common.WeightedTrainer(
        class_weights=common.class_weights(train_df),
        label_smoothing=cand["label_smoothing"],
        model=model,
        args=training_args,
        train_dataset=common.to_dataset(train_df, tok),
        eval_dataset=common.to_dataset(val_df, tok),
        data_collator=DataCollatorWithPadding(tok),
        compute_metrics=common.compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=PATIENCE)],
    )
    trainer.train()
    metrics = trainer.evaluate()
    return {**cand,
            "val_f1": metrics["eval_f1"], "val_macro_f1": metrics["eval_macro_f1"],
            "val_accuracy": metrics["eval_accuracy"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="roberta-base",
                        help="base checkpoint or a dapt.py output dir")
    parser.add_argument("--augmented", action="store_true", help="search on the augmented training data")
    parser.add_argument("--output-dir", type=Path, default=common.RUNS_DIR / "search")
    parser.add_argument("--candidates", type=Path, default=None,
                        help="optional JSON file overriding the built-in candidate list")
    parser.add_argument("--smoke", action="store_true", help="1 tiny candidate, then exit")
    args = parser.parse_args()

    candidates = json.loads(args.candidates.read_text()) if args.candidates else CANDIDATES

    train_path = common.data_dirs(args.augmented)[0] / f"lab-manual-combine-train-{common.SEARCH_SEED}.xlsx"
    train_df, val_df = common.load_split(train_path, common.SEARCH_SEED)
    print(f"search data: train={len(train_df)} val={len(val_df)} (seed {common.SEARCH_SEED} only)")

    if args.smoke:
        candidates = [{"name": "smoke", "lr": 2e-5, "batch_size": 8, "label_smoothing": 0.05, "epochs": 1}]
        train_df, val_df = train_df.head(16), val_df.head(16)
        args.output_dir = common.RUNS_DIR / "smoke-search"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "validation_search.csv"
    done = pd.read_csv(results_path).to_dict("records") if results_path.exists() else []
    done_names = {r["name"] for r in done}

    for cand in candidates:
        if cand["name"] in done_names:
            print(f"[{cand['name']}] already done, skipping")
            continue
        print(f"[{cand['name']}] {cand}")
        result = run_candidate(cand, args, train_df, val_df)
        done.append(result)
        pd.DataFrame(done).to_csv(results_path, index=False)
        print(f"[{cand['name']}] val F1 {result['val_f1']:.4f}  macro {result['val_macro_f1']:.4f}")

    ranked = sorted(done, key=lambda r: (r["val_f1"], r["val_macro_f1"]), reverse=True)
    best = ranked[0]
    selected = {k: best[k] for k in ("name", "lr", "batch_size", "label_smoothing", "epochs")}
    selected.update({"model": args.model, "augmented": args.augmented, "patience": PATIENCE})
    (args.output_dir / "selected_config.json").write_text(json.dumps(selected, indent=2))

    print("\n=== search results (best first) ===")
    for r in ranked:
        print(f"  {r['name']:15} val F1 {r['val_f1']:.4f}  macro {r['val_macro_f1']:.4f}  acc {r['val_accuracy']:.4f}")
    print(f"\nfrozen: {selected} -> {args.output_dir / 'selected_config.json'}")


if __name__ == "__main__":
    main()
