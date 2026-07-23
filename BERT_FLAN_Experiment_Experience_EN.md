# BERT and FLAN-T5 Experiment Experience Summary

## 1. Document Scope

This document consolidates the experiment experience from the COMP9444 FOMC three-class classification task for BERT-family models and FLAN-T5. It is an additional experiment record and handoff document. BERT, FinBERT and RoBERTa are assigned to other group members, so the BERT-family experiments in this document should not be added to the LSTM/BiLSTM member's final fair-comparison table.

The task labels are:

| Label | Meaning |
|---:|---|
| 0 | Dovish |
| 1 | Hawkish |
| 2 | Neutral |

The main conclusion is that the useful performance comes from a pretrained language model followed by task fine-tuning, not simply from downloading a checkpoint and running a few seeds. The experiments also used fixed validation selection, early stopping, best-checkpoint restoration and evaluation on multiple official test seeds.

## 2. Shared Experimental Protocol

Both experiment groups followed the same basic protocol:

1. Use the shared original FOMC train/test Excel data.
2. Search configurations on the train/validation split for seed `5768`.
3. Freeze the configuration, then evaluate on official test seeds `5768`, `78516` and `944601`.
4. Do not use test results for tuning or model selection.
5. Record accuracy, weighted F1, macro F1 and per-class F1.
6. Use dropout, learning-rate control, weight decay, gradient clipping, early stopping and best-checkpoint restoration during training.

The seeds are not switches that automatically produce a high score. They are part of the data-split, randomisation and final stability-evaluation protocol.

## 3. Data Processing Without Augmentation

### 3.1 Original Data Loading

The data comes from the `lab-manual-combine-train-*.xlsx` and `lab-manual-combine-test-*.xlsx` files for each seed. The code reads the `sentence` and `label` fields directly from Excel and keeps the original sample index and year for auditing.

### 3.2 Light Text Cleaning Only

The no-augmentation version performs only whitespace normalisation: it strips leading and trailing whitespace and collapses repeated whitespace into one space. It does not use back translation, synonym replacement, directional word swaps, masking, duplicated training samples or changed labels.

For FLAN-T5, there is no additional manual lowercasing or punctuation removal. The sentence is passed to FLAN-T5's SentencePiece tokenizer. The tokenizer performs subword tokenisation, while the batch collator performs maximum-length truncation, dynamic padding and `attention_mask` generation.

### 3.3 Fixed Splits and Test Data

The seed `5768` training data is divided into a training part and a validation part, and the shared split files are saved. The official test workbooks remain unchanged until the configuration is frozen and final evaluation begins. Therefore, the earlier non-augmented FLAN results do not include the group member's data augmentation work.

## 4. BERT-Family Experiments

### 4.1 Models Used

The exploratory record tested:

- FinBERT: `ProsusAI/finbert`;
- DeBERTa: `microsoft/deberta-v3-base`.

The common approach was to use a pretrained tokenizer and Transformer representation, then fine-tune the model for the FOMC Dovish/Hawkish/Neutral classification task.

### 4.2 Additional Processing and Training

The BERT-family experiment did include processing, but these steps are standard downstream classification training rather than data augmentation:

- Use the tokenizer for truncation, padding and attention-mask handling;
- Create a new classification head for the three FOMC labels;
- Replace the original FinBERT head because its original labels were sentiment labels rather than Dovish/Hawkish/Neutral labels;
- Reinitialise and train the DeBERTa classifier and pooler because they did not contain task-specific parameters;
- Search dropout, learning rate, scheduler, warm-up, weight decay, label smoothing, loss, batch size and maximum length;
- Use early stopping, gradient clipping and best-checkpoint restoration.

No data augmentation, additional manual annotation, label modification or augmented test set was used.

### 4.3 Exact FinBERT Pipeline

The FinBERT run was a standard supervised downstream fine-tuning experiment, not a zero-shot prediction experiment. The sequence was:

1. Load the local or downloaded `ProsusAI/finbert` checkpoint and its matching tokenizer.
2. Read the shared FOMC sentence/label records and keep the original sample index and year for audit purposes.
3. Use the fixed seed-`5768` train/validation split for all configuration search candidates.
4. Tokenise each sentence with truncation and padding, and pass the attention mask to the Transformer.
5. Replace the original sentiment head with a new three-class head for Dovish, Hawkish and Neutral.
6. Fine-tune the pretrained model on the FOMC training partition using the candidate training configuration.
7. Evaluate on the untouched validation partition after each epoch.
8. Restore the best checkpoint using the validation result and stop early when validation stopped improving.

The original exploratory selection key was validation weighted F1 first, followed by macro F1 and parameter count. Accuracy was recorded for every candidate, but it was not the first selection key in that run. This is important because the later FLAN experiments used validation accuracy as the primary selection metric.

### 4.4 FinBERT Search History

Fifteen controlled FinBERT candidates were run on seed `5768` validation. Each candidate changed a limited part of the training configuration so that the effect of the change could be observed:

| Candidate | Main change | Weighted F1 | Accuracy |
|---:|---|---:|---:|
| 1 | control | 0.6613 | 0.6562 |
| 2 | learning rate `1e-5` | 0.6259 | 0.6247 |
| 3 | learning rate `3e-5` | 0.6816 | 0.6798 |
| 4 | dropout `0.2` | 0.6412 | 0.6352 |
| 5 | dropout `0.3` | 0.6237 | 0.6168 |
| 6 | dropout `0.2`, learning rate `1e-5` | 0.6114 | 0.6037 |
| 7 | dropout `0.2`, learning rate `3e-5` | 0.6748 | 0.6693 |
| 8 | cosine schedule | 0.6585 | 0.6562 |
| 9 | warm-up ratio `0.2` | 0.6412 | 0.6352 |
| 10 | weight decay `0` | 0.6615 | 0.6588 |
| 11 | weight decay `0.05` | 0.6613 | 0.6562 |
| 12 | label smoothing `0.05` | 0.6661 | 0.6614 |
| 13 | balanced loss | 0.6696 | 0.6667 |
| 14 | maximum length `128` | 0.6594 | 0.6562 |
| 15 | batch size `8`, longer training | **0.6867** | **0.6824** |

The best observed candidate was candidate 15. The result suggests that batch size and sufficient training time mattered, but it does not prove that the setting generalises to the official test distribution because the three-seed test stage was not completed.

### 4.5 Exact DeBERTa Pipeline and Search Status

The DeBERTa experiment followed the same data and training pipeline, but used `microsoft/deberta-v3-base`. Because the base checkpoint did not contain a task-specific FOMC classifier, the classifier and pooler were newly initialised before downstream training. The run tested the control configuration and then changed learning rate and dropout in controlled candidates.

The run was intentionally stopped during candidate 6 after the overlap with the assigned BERT-family work was confirmed:

| Candidate | Main change | Weighted F1 | Accuracy | Status |
|---:|---|---:|---:|---|
| 1 | control | **0.7373** | **0.7349** | Completed validation |
| 2 | learning rate `1e-5` | 0.6679 | 0.6640 | Completed validation |
| 3 | learning rate `3e-5` | 0.7288 | 0.7270 | Completed validation |
| 4 | dropout `0.2` | 0.6980 | 0.6903 | Completed validation |
| 5 | dropout `0.3` | 0.3203 | 0.4882 | Completed validation |
| 6 | dropout `0.2`, learning rate `1e-5` | 0.5930 | 0.5827 | Run stopped after this result |

The `0.7349` DeBERTa accuracy is therefore a seed-`5768` validation result. It is useful evidence that the model family may be strong, but it is not a final score because no frozen three-seed test evaluation was completed.

### 4.6 Artifact Status and Reproducibility Boundary

The exploratory FinBERT/DeBERTa training script and downloaded checkpoints were removed after the model-family overlap was identified. The current repository keeps the experiment history and this handoff document, but it does not contain the original BERT training script or BERT checkpoint files. The `transformers` package may still be installed in the environment because it is also used by the group's independent Transformer experiments; that does not mean that FinBERT or DeBERTa weights are part of this work package.

This boundary is deliberate: the records preserve what was tested and what was learned, while the final LSTM/BiLSTM work does not claim ownership of another member's BERT-family implementation.

### 4.7 BERT-Family Validation Results

The following results are from seed `5768` validation only. They are not final results from the three official test seeds.

| Model | Best candidate | Weighted F1 | Accuracy | Status |
|---|---:|---:|---:|---|
| FinBERT | candidate 15 | 0.6867 | 0.6824 | Exploratory result |
| DeBERTa | candidate 1 control | 0.7373 | 0.7349 | Exploratory result |

The DeBERTa `73.49%` is validation accuracy. The experiment was stopped during the continued search because the BERT-family direction overlapped with another group member's task. It did not complete the three official test-seed evaluation after configuration freezing. Therefore, `73.49%` must not be presented as the final project generalisation accuracy.

### 4.8 Lessons from the BERT-Family Experiments

1. Pretrained language representations can obtain much stronger validation results than randomly initialised RNNs.
2. The learning rate matters substantially. The default `2e-5` was not always best, and `3e-5` worked better for some candidates.
3. A larger dropout is not automatically more stable. Some DeBERTa candidates dropped sharply after increasing dropout.
4. A pretrained model's original classification head may not match the current labels. A new task head should be created when the label system changes.
5. A high score on one validation seed is only directional evidence. Generalisation should be confirmed with the frozen configuration and multiple test seeds.

## 5. Non-Augmented FLAN-T5 Experiments

### 5.1 Model Structure

FLAN-T5 uses the local `pretrained_models/flan-t5-base` checkpoint. It is not prompted to generate a label. Instead, the model is used as follows:

1. Use the FLAN-T5 encoder to obtain sentence representations;
2. Apply mean pooling to the encoder output;
3. Attach a three-class classification head;
4. Train directly with the Dovish/Hawkish/Neutral labels and classification loss.

The main benefit therefore comes from pretrained language representations and task fine-tuning, not from a generative prompting trick.

### 5.2 v1 Configuration and Results

The v1 search selected the following validation-accuracy configuration:

| Parameter | Value |
|---|---|
| Maximum length | 256 |
| Pooling | mean |
| Classification head | linear |
| Dropout | 0.1 |
| Learning rate | `3e-5` |
| Weight decay | 0.01 |
| Batch size | 4 |
| Gradient accumulation | 2 |
| Maximum epochs | 6 |
| Early-stopping patience | 2 |
| Gradient clipping | 1.0 |

The three official test-seed results were:

| Seed | Accuracy | Weighted F1 |
|---:|---:|---:|
| 5768 | 54.41% | 54.50% |
| 78516 | 61.34% | 61.59% |
| 944601 | 60.29% | 60.84% |
| Mean | **58.68%** | **58.98%** |

### 5.3 v2 Configuration and Results

v2 did not replace the model family. It performed a controlled upgrade of the FLAN-T5 classifier by comparing mean, EOS and attention pooling, linear and MLP heads, learning rates, losses, schedulers, freezing strategies and maximum lengths.

The final validation-selected configuration was `mlp_mean`:

| Parameter | Value |
|---|---|
| Maximum length | 128 |
| Pooling | mean |
| Classification head | MLP head |
| Dropout | 0.2 |
| Encoder learning rate | `2e-5` |
| Head learning rate | `1e-4` |
| Weight decay | 0.01 |
| Scheduler | plateau |
| Maximum epochs | 8 |
| Early-stopping patience | 2 |
| Gradient clipping | 1.0 |
| Data augmentation | false |

On seed `5768` validation, this configuration achieved:

- Accuracy: `69.55%`;
- Weighted F1: `69.31%`;
- Macro F1: `67.32%`.

After freezing the configuration, the three official test-seed results were:

| Seed | Accuracy | Weighted F1 | Macro F1 |
|---:|---:|---:|---:|
| 5768 | 63.66% | 63.24% | 61.05% |
| 78516 | 64.08% | 64.28% | 62.08% |
| 944601 | 60.92% | 61.46% | 59.51% |
| Mean | **62.89%** | **62.99%** | **60.88%** |

### 5.4 Lessons from the FLAN Experiments

1. The FLAN-T5 v2 MLP head was more effective than the v2 control on validation, but the validation improvement did not translate into a test mean above 72% accuracy.
2. A maximum length of `128` was better than `192/256`, suggesting that most sentences in this dataset do not require a longer context.
3. EOS pooling was unstable on this task, while mean pooling was more reliable.
4. Increasing the classification-head learning rate helped, but was less effective than using the MLP head.
5. Class weighting and focal loss did not improve overall accuracy. Improving minority-class metrics alone is not enough if it sacrifices total accuracy.
6. These FLAN results were obtained from the original training set without augmentation and must not be mixed with the later augmented version.

## 6. Why BERT-Family Validation Looks Higher Than FLAN

The current records support three careful explanations:

1. The DeBERTa `73.49%` from the BERT-family experiment is a single validation-seed result, not a final test mean;
2. FLAN v2 reached `69.55%` validation accuracy but achieved a `62.89%` official test mean, so there was a validation-to-test drop;
3. The two groups use different pretraining objectives, tokenizers, classification heads and training hyperparameters. Their difference cannot be explained only by downloading a model and running seeds.

The safest statement is therefore: the BERT/DeBERTa exploration showed a strong validation direction but did not complete a fair official test loop; the non-augmented FLAN-T5 v2 completed the three-seed test evaluation with a mean accuracy of about `62.89%`, and it did not reach the earlier DeBERTa validation value of `73.49%`.

## 7. Boundary with the Augmented Version

The FLAN v1/v2 results in this document are Version 01: the original cleaned data without augmentation. Version 02 loads the group member's augmented training data while preserving the original validation/test protocol. The augmented version should be reported separately by checking:

- Whether the three-seed mean accuracy improves;
- Whether weighted F1 and macro F1 improve;
- Whether at least two of the three seeds exceed the no-augmentation version;
- Whether minority-class F1 shows a trade-off;
- Whether the training cost increases substantially.

Before the augmented results are complete, it should not be assumed that augmentation will necessarily improve accuracy.

## 8. Information Sources

- `BERT_Exploratory_Experiment_History_EN.md`
- `Nonpaper_Transformer_Experiment_History_EN.md`
- `Nonpaper_Transformer_Experiment_Summary_CN.md`
- `notebooks/LSTM_BiLSTM_TCN_Complete_EN.ipynb`
- `notebooks/LSTM_BiLSTM_TCN_Complete_CN.ipynb`
- `results/flan_t5_nonpaper_v1/validation_search/validation_search.csv`
- `results/flan_t5_nonpaper_v1/final/flan_t5_summary.csv`
- `results/flan_t5_nonpaper_v2/validation_search/validation_search.csv`
- `results/flan_t5_nonpaper_v2/validation_search/selected_config.json`
- `results/flan_t5_nonpaper_v2/final/flan_t5_v2_summary.csv`
