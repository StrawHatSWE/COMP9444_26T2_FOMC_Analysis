"""Step 4 - Soft-vote ensemble of restart models, per seed.

Averages softmax probabilities across the restart models of ONE seed and
evaluates on that seed's raw test file. Models are never ensembled across
seeds: seed A's training data legitimately contains sentences that sit in
seed B's test file, so a cross-seed ensemble would leak. Restarts of the
same seed share an identical train/test boundary, which is what makes this
ensemble legal.

  python ensemble.py --models-dir runs/final
  python ensemble.py --models-dir runs/smoke-final --smoke
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

import common


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, default=common.RUNS_DIR / "final")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--smoke", action="store_true", help="evaluate on 8 test rows only")
    args = parser.parse_args()

    per_seed = {}
    for seed in common.FINAL_SEEDS:
        model_dirs = sorted(d for d in args.models_dir.glob(f"model-{seed}-r*") if d.is_dir())
        if not model_dirs:
            continue
        test_df = pd.read_excel(common.data_dirs(False)[1] / f"lab-manual-combine-test-{seed}.xlsx")
        if args.smoke:
            test_df = test_df.head(8)
        sentences = test_df["sentence"].astype(str).tolist()
        labels = test_df["label"].to_numpy()

        probs = np.mean(
            [common.predict_proba(d, sentences, device=args.device) for d in model_dirs], axis=0)
        preds = probs.argmax(axis=1)

        per_seed[seed] = {
            "n_models": len(model_dirs),
            "accuracy": float(accuracy_score(labels, preds)),
            "f1": float(f1_score(labels, preds, average="weighted", zero_division=0)),
            "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        }
        print(f"[seed {seed}] ensemble of {len(model_dirs)}: "
              f"acc {per_seed[seed]['accuracy']:.4f}  F1 {per_seed[seed]['f1']:.4f}  "
              f"macro {per_seed[seed]['macro_f1']:.4f}")

    if not per_seed:
        raise SystemExit(f"no model-<seed>-r* directories found under {args.models_dir}")

    f1s = [m["f1"] for m in per_seed.values()]
    accs = [m["accuracy"] for m in per_seed.values()]
    summary = {
        "per_seed": per_seed,
        "mean_f1": float(np.mean(f1s)), "std_f1": float(np.std(f1s)),
        "mean_accuracy": float(np.mean(accs)), "std_accuracy": float(np.std(accs)),
    }
    out = args.models_dir / "results_ensemble.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nensemble over {len(per_seed)} seeds: "
          f"F1 {summary['mean_f1']:.4f} +/- {summary['std_f1']:.4f}  "
          f"acc {summary['mean_accuracy']:.4f} +/- {summary['std_accuracy']:.4f}")
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
