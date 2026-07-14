# COMP9444 FOMC Five-Member Work Allocation and Unified Experimental Plan

**Task:** Classify FOMC sentences as Dovish, Hawkish or Neutral  
**Team size:** 5 members

## 1. Overall Structure

Recommended structure:

> One shared Benchmark / Infrastructure work package and four model-family work packages.

The five members will be responsible for:

1. baselines, data splits, shared evaluation and plotting;
2. LSTM / BiLSTM;
3. BERT;
4. FinBERT;
5. RoBERTa and the sentence-splitting extension.

The final result table will contain more than five rows. The five work packages can jointly produce approximately 10–12 main results and several ablations.

Recommended hierarchy:

```text
Level 0: Majority / Stratified Random
Level 1: TF-IDF + Logistic Regression / Linear SVM
Level 2: Shah / Gorodnichenko Rule-Based
Level 3: LSTM / BiLSTM
Level 4: BERT-base
Level 5: FinBERT
Level 6: RoBERTa-base
External Reference: published results / released FOMC-RoBERTa
```

This route answers:

- how strong non-learning methods are;
- whether conventional ML beats minimum baselines;
- whether expert rules beat statistical learning;
- whether RNNs learn useful context;
- how much Transformers improve over RNNs;
- whether finance-domain pretraining beats general pretraining;
- whether RoBERTa reproduces the paper's advantage;
- whether fine-tuning beats zero-shot inference or a frozen encoder;
- whether sentence splitting helps.

## 2. Shared Data Protocol

### 2.1 Main Data

All main experiments use `Combined, unsplit`:

```text
lab-manual-combine-train-5768.xlsx
lab-manual-combine-test-5768.xlsx
lab-manual-combine-train-78516.xlsx
lab-manual-combine-test-78516.xlsx
lab-manual-combine-train-944601.xlsx
lab-manual-combine-test-944601.xlsx
```

Fixed labels:

```text
0 = Dovish
1 = Hawkish
2 = Neutral
```

Seed `5768` may be used for development. Final results must run:

```text
5768
78516
944601
```

### 2.2 Validation Split

Member A generates fixed validation splits:

```python
train_test_split(
    training_indices,
    test_size=0.20,
    random_state=seed,
    stratify=labels
)
```

Save:

```text
data/splits/5768_train.csv
data/splits/5768_val.csv
data/splits/78516_train.csv
data/splits/78516_val.csv
data/splits/944601_train.csv
data/splits/944601_val.csv
```

Every model must use exactly the same train, validation and test samples.

### 2.3 Shared Basic Processing

All members must:

- preserve the original sentence;
- strip unnecessary leading and trailing whitespace;
- normalise line breaks and repeated spaces;
- use a shared sample ID;
- not delete samples independently;
- not change labels;
- not deduplicate the main experiment independently;
- run the same exact and normalised train/test overlap audit;
- record year, seed and source file consistently.

### 2.4 Model-Specific Tokenisation

A shared dataset does not require one tokenizer for every model.

| Model | Feature or tokenizer |
|---|---|
| TF-IDF | sklearn word n-grams |
| Rule-Based | official dictionary rules |
| LSTM / BiLSTM | word-level tokenizer |
| BERT | BERT tokenizer |
| FinBERT | tokenizer of the selected checkpoint |
| RoBERTa | RoBERTa tokenizer |

Shared Transformer settings:

```text
max_length = 256
truncation = True
dynamic padding
```

EDA must report the actual truncation rate.

### 2.5 Test Discipline

All members must follow:

- do not use test data to choose learning rate;
- do not use test data to choose batch size;
- do not use test data to choose epoch count;
- do not use test data to select the best checkpoint;
- select checkpoints by validation weighted F1;
- freeze hyperparameters before evaluating all three test sets.

## 3. Member A: Baselines, Data and Shared Evaluation

### Required Baselines

| ID | Baseline | Trained | Pretrained |
|---|---|---:|---:|
| B0 | Majority Class | No | No |
| B1 | Stratified Random | No | No |
| B2 | TF-IDF + Logistic Regression | Yes | No |
| B3 | TF-IDF + Linear SVM | Yes | No |
| B4 | Shah / Gorodnichenko Rule-Based | No | No |

Optional:

| ID | Baseline | Owner |
|---|---|---|
| B5 | FinBERT Sentiment Zero-Shot Mapping | Member D |

### Technical Path

**Majority:** always predicts the most frequent training class.  
**Stratified Random:** samples according to the training distribution with a fixed seed.  
**TF-IDF + Logistic Regression:**

```text
ngram_range: (1,1), (1,2)
min_df: 1, 2
sublinear_tf: True
C: 0.1, 1, 10
class_weight: None
```

**TF-IDF + Linear SVM:**

```text
C: 0.1, 1, 10
class_weight: None
```

**Rule-Based:** reproduce the Shah paper's A1/A2/B1/B2 dictionaries, Group C negation, Hawkish/Dovish reversal and Neutral fallback.

### Shared Code

```text
src/data_loader.py
src/make_splits.py
src/evaluate.py
src/plot_results.py
src/aggregate_results.py
```

### Member A Deliverables

For every baseline and seed:

```text
metrics.json
predictions.csv
classification_report.csv
confusion_matrix_raw.csv
confusion_matrix_normalized.csv
confusion_matrix_normalized.png
```

Member A also owns:

- the aggregate results table;
- the all-model weighted-F1 figure;
- the macro-F1 figure;
- the per-class-F1 figure;
- the performance-vs-cost figure;
- best-baseline vs best-Transformer confusion matrices.

### Literature Responsibility

Summarise Loughran & McDonald (2011) and review Shah's Rule-Based section.

## 4. Member B: LSTM / BiLSTM

### Main Models

```text
M1 = LSTM
M2 = BiLSTM
```

### Main Path

```text
word-level tokenizer
randomly initialised embedding
single-layer LSTM / BiLSTM
softmax output
cross-entropy loss
```

Recommended base configuration:

```text
vocabulary size = 10,000
max sequence length = 200 words
padding = post
embedding dimension = 100 or 200
```

### Optional Pretraining

Only for the best RNN:

```text
BiLSTM + GloVe 100d
```

Compare:

```text
Random Embedding
vs
Static Pretrained Embedding
```

### Hyperparameters

```text
embedding_dim: 100, 200
hidden_size: 64, 128
dropout: 0.3, 0.5
learning_rate: 1e-3, 3e-4
batch_size: 16, 32
max_epochs: 20
early_stopping_patience: 3
```

Tune on seed 5768, freeze, then run the other two seeds.

### Deliverables

For every model and seed:

```text
config.json
metrics.json
predictions.csv
classification_report.csv
training_history.csv
learning_curve.png
confusion_matrix_raw.csv
confusion_matrix_normalized.csv
confusion_matrix_normalized.png
```

Figures:

- LSTM vs BiLSTM weighted F1;
- LSTM vs BiLSTM macro F1;
- both learning curves;
- best-RNN confusion matrix;
- optional Random Embedding vs GloVe.

Error analysis must include at least 10 cases involving negation, mixed tone, long sentences, inflation/employment direction and Neutral errors.

### Literature Responsibility

Summarise Hansen & McMahon (2016).

## 5. Member C: BERT

### Main Models

```text
M3a = BERT Frozen Encoder
M3b = BERT Full Fine-Tuning
```

### Technical Path

**Frozen BERT:**

```text
pretrained BERT encoder frozen
only classification head trained
```

**Fine-Tuned BERT:**

```text
bert-base-uncased
all layers fine-tuned on FOMC
```

### Shared Training Settings

```text
max_length = 256
optimizer = AdamW
weight_decay = 0.01
warmup_ratio = 0.1
max_epochs = 10
early_stopping_patience = 2
checkpoint metric = validation weighted F1
```

### Hyperparameter Grid

Test only on seed 5768:

| Learning rate | Batch size |
|---:|---:|
| 1e-5 | 8 |
| 2e-5 | 8 |
| 5e-5 | 8 |
| 1e-5 | 16 |
| 2e-5 | 16 |
| 5e-5 | 16 |

### Deliverables

```text
results/bert_frozen/
results/bert_finetuned/
```

For every seed:

```text
config.json
metrics.json
predictions.csv
classification_report.csv
training_history.csv
learning_curve.png
confusion_matrix_raw.csv
confusion_matrix_normalized.csv
confusion_matrix_normalized.png
```

Figures:

- Frozen vs Fine-Tuned BERT;
- BERT learning curve;
- BERT confusion matrix;
- BERT per-class F1.

### Literature Responsibility

Summarise MONOPOLY (2022). Only literature review, limitations and future work are required; MPCNet does not need to be implemented.

## 6. Member D: FinBERT

### Main Experiments

```text
B5 = FinBERT Sentiment Zero-Shot Mapping
M4 = Finance-domain BERT Fine-Tuned on FOMC
```

### Path One: Zero-Shot

```text
Positive → Dovish
Negative → Hawkish
Neutral → Neutral
```

No FOMC training. This is a baseline.

### Path Two: Finance-Domain Fine-Tuning

```text
finance-domain pretrained BERT checkpoint
→ FOMC three-class head
→ full fine-tuning
```

It must match Member C as closely as possible:

- same train/validation/test samples;
- same max length;
- same tuning budget;
- same early stopping;
- same evaluator.

This allows a fair answer to:

> Does finance-domain pretraining outperform general BERT?

### Checkpoint Documentation

Record:

- model repository;
- model revision;
- tokenizer;
- whether it is sentiment-fine-tuned;
- whether it is only domain-pretrained;
- whether it is subsequently fine-tuned on FOMC.

Do not write only “FinBERT”.

### Deliverables

```text
results/finbert_zero_shot/
results/finbert_fomc_finetuned/
```

Figures:

- Zero-Shot vs Fine-Tuned;
- BERT vs FinBERT weighted F1;
- BERT vs FinBERT per-class F1;
- FinBERT confusion matrix;
- FinBERT learning curve.

### Literature Responsibility

Summarise Araci (2019), FinBERT.

## 7. Member E: RoBERTa and Extensions

### Main Model

```text
M5 = RoBERTa-base
```

### Main Path

```text
roberta-base
→ three-class classification head
→ full fine-tuning on Combined
```

Use the same train/validation/test samples, max length, tuning budget, checkpoint metric and evaluator as BERT and FinBERT.

### Extension Priority

First priority:

```text
RoBERTa-base on Combined
vs
RoBERTa-base on Combined-S
```

Second priority, choose one if resources permit:

```text
RoBERTa-large
```

or:

```text
FLANG-RoBERTa-base
```

### Deliverables

```text
results/roberta_base_combined/
results/roberta_base_combined_split/
```

Optional:

```text
results/roberta_large/
```

Figures:

- RoBERTa vs BERT vs FinBERT;
- Combined vs Combined-S;
- RoBERTa confusion matrix;
- RoBERTa learning curve;
- difference from Shah paper results.

### Literature Responsibility

Summarise Shah et al. (2023). Every member must read at least its Dataset, Models, Results, FinBERT appendix and Annotation Guide.

## 8. Unified Treatment of Pretrained Models

### Valid Starting Points

| Type | Use |
|---|---|
| BERT-base | fine-tune on FOMC |
| Finance-domain BERT | fine-tune on FOMC |
| RoBERTa-base | fine-tune on FOMC |
| GloVe | RNN embedding ablation |

### Not Team-Trained Models

A model already trained by the authors on FOMC data, such as:

```text
FOMC-RoBERTa
```

must be labelled:

```text
External Released Model
```

It may be used for demonstrations, sanity checks, a few prediction examples and external reference. It must not appear in the main fair-comparison table as `Our RoBERTa model`.

### Published Scores

Use a `Paper Reference` column:

| Model | Our weighted F1 | Paper weighted F1 | Difference |
|---|---:|---:|---:|
| Rule-Based |  | 0.4966 |  |
| BiLSTM |  | 0.5387 |  |
| BERT-base |  | 0.6310 |  |
| FinBERT |  | 0.6325 |  |
| RoBERTa-base |  | 0.6755 |  |

Published scores are references, not exact mandatory targets.

## 9. Unified Metrics

Primary metric:

```text
Weighted F1 mean ± standard deviation over 3 seeds
```

Also report:

```text
Macro F1 mean ± std
Accuracy mean ± std
Dovish precision / recall / F1
Hawkish precision / recall / F1
Neutral precision / recall / F1
Training time
Inference time, optional
Parameter count
```

## 10. Unified Result Files

Every model directory:

```text
results/<model_name>/
├── config.json
├── seed_5768/
│   ├── metrics.json
│   ├── predictions.csv
│   ├── classification_report.csv
│   ├── confusion_matrix_raw.csv
│   ├── confusion_matrix_normalized.csv
│   ├── confusion_matrix_normalized.png
│   ├── training_history.csv
│   └── learning_curve.png
├── seed_78516/
├── seed_944601/
└── aggregate_metrics.json
```

`predictions.csv`:

```text
sample_id
year
sentence
true_label
predicted_label
prob_dovish
prob_hawkish
prob_neutral
correct
seed
model_name
```

`config.json` must record:

```text
model_name
pretrained_checkpoint
checkpoint_revision
training_type
tokenizer
max_length
learning_rate
batch_size
epochs
best_epoch
dropout
weight_decay
random_seed
train_split_file
validation_split_file
test_split_file
library_versions
hardware
```

## 11. Unified Figures

All members use Member A's plotting script.

### Confusion Matrix

```text
Label order: Dovish, Hawkish, Neutral
Rows: True label
Columns: Predicted label
Normalisation: by true class
Display: percentage
```

### Learning Curve

Only for trained models:

```text
Training Loss
Validation Loss
Validation Weighted F1
Best Epoch marker
```

### Per-Class F1

Fixed order:

```text
Dovish
Hawkish
Neutral
```

## 12. Final Group Figures

Member A generates:

1. weighted F1 across all models with three-seed error bars;
2. macro F1 across all models;
3. per-class F1 for the best conventional baseline, BiLSTM, BERT, FinBERT and RoBERTa;
4. performance vs training time or parameter count;
5. best-baseline vs best-Transformer confusion matrices;
6. RoBERTa Combined vs Combined-S.

## 13. Final Tables

Main table, using the same `Combined unsplit` data:

| Model | Pretraining | Weighted F1 | Macro F1 | Accuracy | Dovish F1 | Hawkish F1 | Neutral F1 | Paper F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|

Ablation table:

| Experiment | Variant A | Variant B | Difference |
|---|---|---|---:|
| Static pretraining | BiLSTM Random | BiLSTM GloVe |  |
| Fine-tuning | BERT Frozen | BERT Fine-Tuned |  |
| Domain pretraining | BERT | FinBERT |  |
| Task adaptation | FinBERT Zero-Shot | FinBERT Fine-Tuned |  |
| Preprocessing | RoBERTa Combined | RoBERTa Combined-S |  |
| Model scale, optional | RoBERTa-base | RoBERTa-large |  |

## 14. Required Written Contribution from Every Member

### Literature Summary

```text
Paper problem
Dataset
Method
Main finding
Limitation
Relevance to our project
```

### Model Method

```text
Why selected
Architecture
Pretrained checkpoint
Data preprocessing
Hyperparameter search
Training procedure
Checkpoint selection
```

### Result Analysis

```text
Main result
Comparison with baselines
Comparison with paper
Per-class behaviour
Strengths
Weaknesses
Five to ten representative errors
```

## 15. Execution Order

### Stage 1: Protocol and Infrastructure

Member A completes data loader, splits, evaluator, result schema and plotting.  
Members B–E confirm checkpoints, create code skeletons and complete literature summaries.

### Stage 2: Baselines and Smoke Tests

Member A runs all lightweight baselines.  
Members B–E each complete one working run on seed 5768.

### Stage 3: Controlled Tuning

Use only:

```text
seed 5768 train + validation
```

### Stage 4: Freeze Hyperparameters

After freezing, run:

```text
5768
78516
944601
```

### Stage 5: Aggregate Results

Member A collects all `metrics.json` and `predictions.csv` files and generates the final tables and figures.

### Stage 6: Extensions

Only after the main table is complete:

- BiLSTM + GloVe;
- Combined-S;
- RoBERTa-large or FLANG-RoBERTa;
- external released-model demonstration.

## 16. Expected Final Output

### Baselines

1. Majority
2. Stratified Random
3. TF-IDF + Logistic Regression
4. TF-IDF + Linear SVM
5. Paper Rule-Based
6. FinBERT Sentiment Zero-Shot

### Main Models

7. LSTM
8. BiLSTM
9. BERT Frozen
10. BERT Fine-Tuned
11. FinBERT Fine-Tuned
12. RoBERTa-base

### Optional Experiments

13. BiLSTM + GloVe
14. RoBERTa Combined-S
15. RoBERTa-large or FLANG-RoBERTa
16. External FOMC-RoBERTa demonstration

> Five members own five work packages and jointly produce approximately 12 main results and 2–4 ablations. All experiments share the same data, validation splits, test discipline, metrics, result format and plotting standards. Pretraining is evaluated only through controlled comparisons where it is technically meaningful, rather than requiring every member to run every checkpoint.
