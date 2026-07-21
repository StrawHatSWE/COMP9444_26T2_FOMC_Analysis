Got it. Here's the data augmentation analysis in English:
 
---
 
## Current Data Profile
 
| Dimension | Status |
|---|---|
| Training samples | ~1,522 per seed (after 80/20 split) |
| Class distribution | Neutral 48.7%, Dovish 26.1%, Hawkish 25.2% (severe imbalance) |
| Sentence length | Mean 32 words, median 29, range 2–181 |
| Duplicates | 33–39 intra-train, 12–15 normalized train/test overlap |
| Best model | BiLSTM v2 weighted F1 = 0.5281, Dovish/Hawkish F1 only ~0.40–0.42 |
| Existing augmentation | **None** |
 
**Core issue: tiny dataset + class imbalance + direction signals (Dovish vs Hawkish) are hard to learn.**
 
---
 
## 6 Recommended Data Augmentation Strategies
 
### 1. Back-Translation — Top Priority
 
Translate English → intermediate language → back to English. Preserves stance while varying syntax.
 
```
Original: "Inflation remains elevated, warranting a restrictive stance."
→ German → English: "Inflation stays high, which justifies a restrictive position."
```
 
- **Why it fits**: FOMC stance is determined by directional meaning, which back-translation largely preserves
- **Risk**: domain terms like "accommodative" may mistranslate
- **Plan**: apply 1–2× augmentation to Dovish/Hawkish only (don't bloat the already-dominant Neutral class)
 
 
### 2. Counterfactual / Directional Word Swap — Most Precise
 
Directly targets the model's worst error category: **"inflation/employment direction"** confusion. Build a directional word map and flip key indicators.
 
```
Original (Hawkish): "Inflation pressures are building and labor markets remain tight."
→ Counterfactual (Dovish): "Inflation pressures are easing and labor markets are softening."
```
 
- Directional mapping:
  - `rising/increasing/building/elevated` ↔ `falling/declining/easing/moderating`
  - `tight/strong/robust` ↔ `soft/weak/slack`
  - `tightening/restrictive` ↔ `accommodative/expansionary`
- Directly addresses 3 most frequent error types (direction, negation, mixed tone)
- Generates positive examples for both minority classes simultaneously
 
 
### 3. EDA (Easy Data Augmentation) — Fast, Proven
 
Wei & Zou (2019) four operations, especially effective for short-text classification:
 
| Operation | Rate | Description |
|---|---|---|
| Synonym Replacement (SR) | 10–15% of words | Replace with WordNet synonyms |
| Random Insertion (RI) | 1–2 words/sentence | Insert synonyms of random words |
| Random Swap (RS) | 1–2 swaps/sentence | Swap adjacent words |
| Random Deletion (RD) | 10% of words | Randomly delete words |
 
- Word-level operations, consistent with the current word-level tokenizer
- Target: 2–3× augmentation for Dovish/Hawkish
 
 
### 4. FinBERT-MLM Contextual Word Replacement — Domain-Aware
 
Mask a random 10–15% of non-keyword tokens and let FinBERT predict replacements.
 
```
Original: "The [MASK] outlook remains uncertain amid [MASK] supply disruptions."
→ FinBERT → "The economic outlook remains uncertain amid ongoing supply disruptions."
```
 
- Replacements come from financial-domain context, much more natural than WordNet
- Faster than back-translation, batch generation possible
- Shares infrastructure with Member D's FinBERT work
 
 
### 5. Class-Weighted Oversampling + EDA Combo — Fix Imbalance
 
Plain oversampling causes overfitting, but **oversample + EDA** is a proven two-step strategy:
 
1. Oversample Dovish/Hawkish to ~600 samples each (random or SMOTE)
2. Apply EDA operations to the oversampled copies for diversity
3. End result: Dovish ~600, Hawkish ~600, Neutral ~743 — balanced
 
 
### 6. Time-Aware Validation Split — Not Augmentation, But Worth Doing
 
Data spans 1996–2022 (27 years). Current random 80/20 split may put same-year sentences in both train/val. Consider:
- Time-based split (e.g., 1996–2018 train, 2019–2022 test)
- Tests robustness to temporal shift
- Fits as an additional ablation in Phase 6
 
 
## Recommended Priority & Implementation Order
 
| Priority | Method | Target | Effort |
|---|---|---|---|
| P0 | Back-Translation | Dovish/Hawkish ×2 | Medium (translation library) |
| P0 | Directional Word Swap | Dovish/Hawkish ×1.5 | Low (build mapping table) |
| P1 | EDA 4 ops | Dovish/Hawkish ×2 | Low |
| P1 | Oversample + EDA | Balance all 3 classes | Low |
| P2 | FinBERT-MLM Replacement | All classes, moderate | Medium (needs FinBERT) |
| P3 | Time-aware split | Robustness validation | Low |
 
---
 
## Key Constraints
 
Per the project plan (Section 2.3), main experiments must not arbitrarily deduplicate or relabel. Augmented data should:
 
1. Only apply to train splits — val/test untouched
2. Clearly label original vs. augmented samples
3. Record augmentation strategy details in `config.json`
4. Run as a separate ablation experiment first (Phase 6), not in the main result table
 
---
 
Which directions do you want to try first? I can implement the two P0 items (back-translation + directional swap) directly, or start with a minimal EDA version to validate the approach.
 