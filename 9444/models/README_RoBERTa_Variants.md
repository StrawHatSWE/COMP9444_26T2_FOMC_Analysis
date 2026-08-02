# RoBERTa enhancement variants

Run these commands from the repository root with the project virtual environment activated.

## 1. RoBERTa + DAPT

Continues masked-language-model pretraining on the classification training portion only, then fine-tunes the adapted checkpoint.

```powershell
python .\9444\models\roberta_dapt.py
```

## 2. RoBERTa + averaged weighted hyperparameter search

Evaluates every learning-rate/batch-size pair using validation weighted-F1 for all three seeds and selects the highest mean. The search runs 6 configurations x 3 seeds before the final runs.

```powershell
python .\9444\models\roberta_average_weighted_hp.py
```

To test the pipeline without the full search:

```powershell
python .\9444\models\roberta_average_weighted_hp.py --skip-hp-search --seeds 5768
```

## 3. RoBERTa + class balancing

Uses inverse-frequency class weights calculated from each seed's training portion in cross-entropy loss.

```powershell
python .\9444\models\roberta_class_balancing.py
```

## 4. RoBERTa + label smoothing

Uses cross-entropy with label smoothing set to `0.1`.

```powershell
python .\9444\models\roberta_label_smoothing.py
```

## 5. RoBERTa + all enhancements

Combines DAPT, averaged validation weighted-F1 hyperparameter search, inverse-frequency class balancing, and `0.1` label smoothing.

```powershell
python .\9444\models\roberta_all_enhancements.py
```

DAPT checkpoints are cached per variant and seed under `9444/results/<variant>/_dapt/`. Results, configurations, histories, classification reports, and aggregate metrics are written under `9444/results/<variant>/`.

All scripts accept `--learning-rate`, `--batch-size`, and `--seeds`. Search-enabled scripts also accept `--skip-hp-search`.
