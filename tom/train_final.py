"""Steps 1 & 3 (final leg) - train the frozen configuration across all seeds.

Reads selected_config.json produced by search.py (CLI flags override), then
per seed trains --restarts independent models (same data split, different
training randomness) for ensemble.py. Evaluation is ALWAYS on the raw test
file of the same seed. Per-model metrics persist immediately, so interrupted
runs resume by re-running the same command.

  python train_final.py                          # uses runs/search/selected_config.json
  python train_final.py --restarts 3
  python train_final.py --smoke                  # functionality check only
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import DataCollatorWithPadding, EarlyStoppingCallback, TrainingArguments

import common


def train_one(seed: int, restart: int, cfg: dict, args, out_dir: Path) -> dict:
    train_path = common.data_dirs(cfg["augmented"])[0] / f"lab-manual-combine-train-{seed}.xlsx"
    train_df, val_df = common.load_split(train_path, seed)
    test_df = pd.read_excel(common.data_dirs(False)[1] / f"lab-manual-combine-test-{seed}.xlsx")

    if args.smoke:
        train_df, val_df, test_df = train_df.head(16), val_df.head(8), test_df.head(8)

    tok, model = common.load_tokenizer_and_model(cfg["model"])

    training_args = TrainingArguments(
        output_dir=str(out_dir / f"checkpoint-{seed}-r{restart}"),
        num_train_epochs=cfg["epochs"],
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=64,
        learning_rate=cfg["lr"],
        warmup_ratio=0.1,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        seed=seed + restart * 1000,          # restart varies training randomness only
        data_seed=seed,                      # data order stays tied to the seed
        report_to=[],
    )
    trainer = common.WeightedTrainer(
        class_weights=common.class_weights(train_df),
        label_smoothing=cfg["label_smoothing"],
        model=model,
        args=training_args,
        train_dataset=common.to_dataset(train_df, tok),
        eval_dataset=common.to_dataset(val_df, tok),
        data_collator=DataCollatorWithPadding(tok),
        compute_metrics=common.compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg["patience"])],
    )
    trainer.train()
    test_metrics = trainer.evaluate(common.to_dataset(test_df, tok), metric_key_prefix="test")

    model_dir = out_dir / f"model-{seed}-r{restart}"
    trainer.save_model(str(model_dir))
    tok.save_pretrained(str(model_dir))
    return {k.replace("test_", ""): v for k, v in test_metrics.items() if k.startswith("test_")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=common.RUNS_DIR / "search" / "selected_config.json")
    parser.add_argument("--model", default=None, help="override config's checkpoint")
    parser.add_argument("--augmented", action="store_true", default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=common.FINAL_SEEDS)
    parser.add_argument("--restarts", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=common.RUNS_DIR / "final")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.config.exists():
        cfg = json.loads(args.config.read_text())
    else:
        print(f"note: {args.config} not found, using built-in defaults")
        cfg = {"model": "roberta-base", "lr": 2e-5, "batch_size": 32,
               "label_smoothing": 0.05, "epochs": 4, "patience": 2, "augmented": False}
    if args.model:
        cfg["model"] = args.model
    if args.augmented is not None:
        cfg["augmented"] = args.augmented
    if args.smoke:
        cfg = {**cfg, "epochs": 1, "batch_size": 8}
        args.seeds, args.restarts = [common.SEARCH_SEED], 1
        args.output_dir = common.RUNS_DIR / "smoke-final"
    print("frozen config:", cfg)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "config_used.json").write_text(json.dumps(cfg, indent=2))

    results: dict[str, dict] = {}
    for seed in args.seeds:
        for restart in range(args.restarts):
            key = f"{seed}-r{restart}"
            metrics_path = args.output_dir / f"metrics-{key}.json"
            if metrics_path.exists():
                results[key] = json.loads(metrics_path.read_text())
                print(f"[{key}] already done, loaded")
                continue
            print(f"[{key}] training...")
            results[key] = train_one(seed, restart, cfg, args, args.output_dir)
            metrics_path.write_text(json.dumps(results[key], indent=2))
            print(f"[{key}] test metrics: { {k: round(v, 4) for k, v in results[key].items() if k in ('accuracy', 'f1', 'macro_f1')} }")

    f1s = [results[k]["f1"] for k in results]
    summary = {"config": cfg, "per_model": results,
               "single_model_mean_f1": float(np.mean(f1s)), "single_model_std_f1": float(np.std(f1s))}
    (args.output_dir / "results.json").write_text(json.dumps(summary, indent=2))
    print(f"\nsingle-model F1 over {len(f1s)} models: {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}")
    print("next: python ensemble.py --models-dir", args.output_dir)


if __name__ == "__main__":
    main()
