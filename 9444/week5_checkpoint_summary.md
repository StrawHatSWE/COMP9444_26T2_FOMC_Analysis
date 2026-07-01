# Week 5 Checkpoint Summary

Project: **003. NLP Classification of FOMC Texts for Market Analysis**

Completed slide deck:

- `COMP9444_FOMC_Checkpoint_Completed.pptx`

Supporting files:

- `Shah_et_al_2023_Trillion_Dollar_Words.pdf`
- `FOMC_dataset_checkpoint.zip`
- `fomc_dataset_analysis.py`

---

## Is the paper reasonable?

Yes. **Shah, Paturi, and Chava (2023), "Trillion Dollar Words: A New Financial Dataset, Task & Market Analysis"** is highly suitable for this checkpoint and for our project because:

1. It is the original/relevant paper for the FOMC hawkish-dovish-neutral classification dataset.
2. It defines the exact NLP task we are working on: sentence-level monetary-policy stance classification.
3. It explains why ordinary positive/negative financial sentiment is not enough for FOMC language.
4. It provides benchmark context, including rule-based, LSTM, BERT-family, finance-domain and RoBERTa-style models.
5. It connects model outputs to market analysis, including Treasury yields, equities and macroeconomic indicators.

Limitations to mention:

- It is very close to the dataset source, so for a broader literature review we should later add other papers such as FinBERT, Loughran-McDonald financial dictionaries, and Hansen-McMahon central-bank communication work.
- The paper's strongest results use large pretrained models; our project scope may need simpler reproducible baselines first.
- Market impact analysis is more complex than sentence classification, so we should separate classification performance from downstream market prediction claims.

Conclusion: **reasonable as the main checkpoint paper; add 2-4 supporting papers later for the final Week 10 presentation/report.**

---

## Slide Deck Coverage

The completed deck covers the required Week 5 checkpoint sections:

- **Introduction / Motivation:** slides 2-3
- **Problem Statement:** slide 3
- **Literature Review:** slides 4-5
- **Dataset Description:** slide 6
- **Basic Dataset Analysis:** slides 7-9
- **Next Steps:** slide 10

---

## Dataset Analysis Results

Dataset package:

```text
FOMC_dataset_checkpoint.zip
```

Files:

```text
lab-manual-combine-train-5768.xlsx
lab-manual-combine-test-5768.xlsx
lab-manual-combine-train-78516.xlsx
lab-manual-combine-test-78516.xlsx
lab-manual-combine-train-944601.xlsx
lab-manual-combine-test-944601.xlsx
```

Columns:

| Column | Meaning |
|---|---|
| index | original sample id |
| sentence | FOMC-related text sentence |
| year | associated year |
| label | 0 = Dovish, 1 = Hawkish, 2 = Neutral |

For each seed:

| Statistic | Value |
|---|---:|
| Train samples | 1,903 |
| Test samples | 476 |
| Total labelled samples | 2,379 |
| Dovish | 625 (26.3%) |
| Hawkish | 600 (25.2%) |
| Neutral | 1,154 (48.5%) |
| Year range | 1996-2022 |
| Mean sentence length | 31.54 words |
| Median sentence length | 29 words |
| Max sentence length | 181 words |
| Empty sentences | 0 |
| Repeated sentence texts | 53 |

Interpretation:

- The task is a **3-class sentence classification** problem.
- The dataset has a moderate class imbalance: Neutral is nearly half the dataset.
- Accuracy alone is not enough; use **macro-F1**, weighted-F1, per-class F1 and confusion matrix.
- The sentences are technical and often context-dependent, so finance/domain-specific models may help.
- The same labelled sample pool appears under three random seeds; report which seed(s) are used.

---

## Suggested Baselines

The project description says there should be three benchmarks/baselines. A reasonable plan:

1. **Majority-class baseline:** always predict Neutral.
2. **Traditional ML baseline:** TF-IDF + Logistic Regression or Linear SVM.
3. **Neural/Transformer baseline:** BERT, FinBERT, RoBERTa, or released FOMC-RoBERTa if allowed.

Recommended metrics:

- Accuracy
- Macro-F1
- Weighted-F1
- Per-class precision/recall/F1
- Confusion matrix

---

## How to Reproduce Dataset Analysis

Run:

```bash
cd /Users/lanxiaoyu/Desktop/9444
/Users/lanxiaoyu/miniconda3/envs/unsw/bin/python3 group/tutweek5/fomc_dataset_analysis.py
```

The script uses only standard-library XML/zip parsing, so it does not need pandas or openpyxl.
