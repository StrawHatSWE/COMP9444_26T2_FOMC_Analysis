# tom — the actual model (RoBERTa FOMC stance classifier)

Implements the five-step accuracy plan on top of the leakage-safe protocol
developed in `finBERT_vulture`. All scripts share `common.py`, which enforces
the two project rules: splits operate on source sentences (never rows), and
no training-side input may overlap any test sentence.

## Pipeline (run order)

| Step | Script | What it does | Output |
|---|---|---|---|
| 1 | (config) | RoBERTa encoder — `--model roberta-base` default, `roberta-large` on GPU | — |
| 2 | `dapt.py` | MLM continued pretraining on unlabeled Fed text (TAPT by default, `--corpus` for full DAPT); corpus test-filtered | `runs/dapt/` checkpoint |
| 3 | `search.py` | Candidate search on seed-5768 validation ONLY; freezes winner | `runs/search/selected_config.json` |
| 3 | `train_final.py` | Frozen config × 3 seeds × `--restarts` N; evaluates on raw test | `runs/final/` models + metrics |
| 4 | `ensemble.py` | Soft-vote across restarts, within seed only (cross-seed = leakage) | `runs/final/results_ensemble.json` |
| 5 | `self_train.py` | Pseudo-label unlabeled corpus at high confidence | `runs/self_train/pseudo_labeled.xlsx` → `train_final.py --extra-train`, round 2 |

Full run:

```
python dapt.py
python search.py --model runs/dapt --augmented
python train_final.py --restarts 3
python ensemble.py
python self_train.py --seed 5768
python train_final.py --extra-train runs/self_train/pseudo_labeled.xlsx --output-dir runs/final-round2 --restarts 3
python ensemble.py --models-dir runs/final-round2
```

Every script has `--smoke` for a functionality check (tiny data, one epoch).
Interrupted runs resume: search and train_final skip completed candidates/models.

## Leakage guarantees

- `common.load_split`: validation from original rows only; augmented/pseudo
  derivatives of validation sentences excluded from training (family split by
  source `(index, year)`).
- `common.all_test_sentences()`: DAPT/self-training corpora filtered against
  every test file (all variants, seeds, augmented sets).
- `train_final.py`/`ensemble.py`: benchmark evaluation always on raw test files.
- `ensemble.py`: refuses cross-seed pooling by construction.
- Search never sees a test file; final seeds run one frozen config.

## Baselines to beat (from finBERT_vulture)

- tuned FinBERT: 0.629 ± 0.011 weighted F1
- published ceiling (RoBERTa-large, Shah et al. 2023): ~0.71–0.74

## How it works

### DAPT via Masked Language Modelling

After parsing arguments we start by adjusting RoBERTa to FOMC data by masking several words from sentences and asking it to fill in the blanks.

After which search.py figures out a winning combination of parameters to run the actual training action on.

train_final.py then does the training and finetuning 3 times per seed.

Each seeds models are then combined in ensemble.py and all three are scored and averaged, making the prediction.
