# COMP9444 FOMC 项目五人分工与统一实验计划

**任务：** 将 FOMC 句子分类为 Dovish、Hawkish 或 Neutral  
**团队规模：** 5 人

## 1. 总体结构

建议采用：

> 1 个共享 Benchmark / Infrastructure 工作包 + 4 个模型家族工作包。

五名成员分别负责：

1. Baseline、数据划分、统一评估和绘图；
2. LSTM / BiLSTM；
3. BERT；
4. FinBERT；
5. RoBERTa 与 sentence-splitting 扩展。

最终主结果不止五行。五个工作包可以共同产生约 10–12 个主结果和若干 ablation。

推荐的技术层级：

```text
Level 0：Majority / Stratified Random
Level 1：TF-IDF + Logistic Regression / Linear SVM
Level 2：Shah / Gorodnichenko Rule-Based
Level 3：LSTM / BiLSTM
Level 4：BERT-base
Level 5：FinBERT
Level 6：RoBERTa-base
External Reference：论文结果 / 作者发布的 FOMC-RoBERTa
```

这条路线用于回答：

- 无学习方法能达到什么水平；
- 传统机器学习是否超过最低 baseline；
- 专家规则是否优于统计学习；
- RNN 是否能学习上下文；
- Transformer 相比 RNN 提升多少；
- 金融领域预训练是否优于通用预训练；
- RoBERTa 是否复现论文优势；
- Fine-tuning 是否优于 zero-shot 或 frozen encoder；
- Sentence splitting 是否有效。

## 2. 全组统一的数据协议

### 2.1 主数据

主实验统一使用 `Combined, unsplit`：

```text
lab-manual-combine-train-5768.xlsx
lab-manual-combine-test-5768.xlsx
lab-manual-combine-train-78516.xlsx
lab-manual-combine-test-78516.xlsx
lab-manual-combine-train-944601.xlsx
lab-manual-combine-test-944601.xlsx
```

标签固定：

```text
0 = Dovish
1 = Hawkish
2 = Neutral
```

开发阶段可以先跑 seed `5768`，最终必须运行：

```text
5768
78516
944601
```

### 2.2 Validation split

成员 A 统一生成固定 validation split：

```python
train_test_split(
    training_indices,
    test_size=0.20,
    random_state=seed,
    stratify=labels
)
```

保存：

```text
data/splits/5768_train.csv
data/splits/5768_val.csv
data/splits/78516_train.csv
data/splits/78516_val.csv
data/splits/944601_train.csv
data/splits/944601_val.csv
```

所有模型必须使用完全相同的 train、validation 和 test 样本。

### 2.3 共享基础处理

全组统一：

- 保留原始 sentence；
- 去除首尾多余空格；
- 统一换行符和连续空格；
- 使用统一 sample ID；
- 不随意删除样本；
- 不修改标签；
- 主实验中不自行去重；
- 统一检查 train/test exact overlap 和 normalized overlap；
- 统一记录 year、seed 和 source file。

### 2.4 模型专属 tokenizer

共享数据不等于所有模型使用相同 tokenizer。

| 模型 | 特征或 tokenizer |
|---|---|
| TF-IDF | sklearn word n-grams |
| Rule-Based | 官方词典规则 |
| LSTM / BiLSTM | word-level tokenizer |
| BERT | BERT tokenizer |
| FinBERT | 对应 checkpoint tokenizer |
| RoBERTa | RoBERTa tokenizer |

Transformer 统一：

```text
max_length = 256
truncation = True
dynamic padding
```

EDA 需要报告实际 truncation rate。

### 2.5 Test discipline

所有成员必须遵守：

- test 不能用于选 learning rate；
- test 不能用于选 batch size；
- test 不能用于选 epoch；
- test 不能用于选最佳 checkpoint；
- 最佳 checkpoint 按 validation weighted F1 选择；
- 超参数冻结后再运行三个 test sets。

## 3. 成员 A：Baseline、数据与统一评估负责人

### 必做 Baseline

| ID | Baseline | 是否训练 | 是否预训练 |
|---|---|---:|---:|
| B0 | Majority Class | 否 | 否 |
| B1 | Stratified Random | 否 | 否 |
| B2 | TF-IDF + Logistic Regression | 是 | 否 |
| B3 | TF-IDF + Linear SVM | 是 | 否 |
| B4 | Shah / Gorodnichenko Rule-Based | 否 | 否 |

可选：

| ID | Baseline | 负责人 |
|---|---|---|
| B5 | FinBERT Sentiment Zero-Shot Mapping | 成员 D |

### 技术路径

**Majority：** 始终预测训练集中最多的类别。  
**Stratified Random：** 按训练集类别比例随机预测并固定 seed。  
**TF-IDF + Logistic Regression：**

```text
ngram_range: (1,1), (1,2)
min_df: 1, 2
sublinear_tf: True
C: 0.1, 1, 10
class_weight: None
```

**TF-IDF + Linear SVM：**

```text
C: 0.1, 1, 10
class_weight: None
```

**Rule-Based：** 尽量复现 Shah 论文中的 A1/A2/B1/B2 词典、C 组否定词、鹰鸽反转和无匹配为 Neutral 的逻辑。

### 共享代码

```text
src/data_loader.py
src/make_splits.py
src/evaluate.py
src/plot_results.py
src/aggregate_results.py
```

### 成员 A 最终产出

每个 baseline、每个 seed：

```text
metrics.json
predictions.csv
classification_report.csv
confusion_matrix_raw.csv
confusion_matrix_normalized.csv
confusion_matrix_normalized.png
```

成员 A 还负责：

- 最终 aggregate result table；
- 所有模型 weighted F1 总图；
- macro F1 总图；
- per-class F1 总图；
- performance vs cost 图；
- 最佳 baseline 与最佳 Transformer confusion matrix。

### 文献职责

总结 Loughran & McDonald (2011)，并补充 Shah 的 Rule-Based 部分。

## 4. 成员 B：LSTM / BiLSTM

### 主模型

```text
M1 = LSTM
M2 = BiLSTM
```

### 主技术路径

```text
word-level tokenizer
randomly initialised embedding
single-layer LSTM / BiLSTM
softmax output
cross-entropy loss
```

建议基础配置：

```text
vocabulary size = 10,000
max sequence length = 200 words
padding = post
embedding dimension = 100 or 200
```

### 可选预训练实验

只对最佳 RNN 额外运行：

```text
BiLSTM + GloVe 100d
```

比较：

```text
Random Embedding
vs
Static Pretrained Embedding
```

### 超参数范围

```text
embedding_dim: 100, 200
hidden_size: 64, 128
dropout: 0.3, 0.5
learning_rate: 1e-3, 3e-4
batch_size: 16, 32
max_epochs: 20
early_stopping_patience: 3
```

只在 seed 5768 调参，随后冻结并运行其他两个 seeds。

### 最终产出

每个模型、每个 seed：

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

图片：

- LSTM vs BiLSTM weighted F1；
- LSTM vs BiLSTM macro F1；
- 两个 learning curves；
- 最佳 RNN confusion matrix；
- 可选 Random Embedding vs GloVe。

错误分析至少 10 个案例，覆盖否定词、混合语气、长句、inflation/employment 方向和 Neutral 误判。

### 文献职责

总结 Hansen & McMahon (2016)。

## 5. 成员 C：BERT

### 主模型

```text
M3a = BERT Frozen Encoder
M3b = BERT Full Fine-Tuning
```

### 技术路径

**Frozen BERT：**

```text
pretrained BERT encoder frozen
only classification head trained
```

**Fine-Tuned BERT：**

```text
bert-base-uncased
all layers fine-tuned on FOMC
```

### 统一训练参数

```text
max_length = 256
optimizer = AdamW
weight_decay = 0.01
warmup_ratio = 0.1
max_epochs = 10
early_stopping_patience = 2
checkpoint metric = validation weighted F1
```

### 超参数网格

只在 seed 5768 测试：

| Learning rate | Batch size |
|---:|---:|
| 1e-5 | 8 |
| 2e-5 | 8 |
| 5e-5 | 8 |
| 1e-5 | 16 |
| 2e-5 | 16 |
| 5e-5 | 16 |

### 最终产出

```text
results/bert_frozen/
results/bert_finetuned/
```

每个 seed 输出：

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

图片：

- Frozen vs Fine-Tuned；
- BERT learning curve；
- BERT confusion matrix；
- BERT per-class F1。

### 文献职责

总结 MONOPOLY (2022)。只做 literature review、limitation 和 future work，不实现 MPCNet。

## 6. 成员 D：FinBERT

### 主实验

```text
B5 = FinBERT Sentiment Zero-Shot Mapping
M4 = Finance-domain BERT Fine-Tuned on FOMC
```

### 路径一：Zero-Shot

```text
Positive → Dovish
Negative → Hawkish
Neutral → Neutral
```

不在 FOMC 上训练，属于 baseline。

### 路径二：Finance-domain Fine-Tuning

```text
finance-domain pretrained BERT checkpoint
→ FOMC three-class head
→ full fine-tuning
```

必须尽量与成员 C 保持一致：

- 相同 train/val/test；
- 相同 max_length；
- 相同调参预算；
- 相同 early stopping；
- 相同 evaluator。

这样才能公平回答：

> 金融领域预训练是否优于通用 BERT？

### Checkpoint 记录要求

必须记录：

- model repository；
- model revision；
- tokenizer；
- 是否 sentiment-fine-tuned；
- 是否仅 domain-pretrained；
- 是否再次在 FOMC 上 fine-tune。

不能只写“FinBERT”。

### 最终产出

```text
results/finbert_zero_shot/
results/finbert_fomc_finetuned/
```

图片：

- Zero-Shot vs Fine-Tuned；
- BERT vs FinBERT weighted F1；
- BERT vs FinBERT per-class F1；
- FinBERT confusion matrix；
- FinBERT learning curve。

### 文献职责

总结 Araci (2019) FinBERT。

## 7. 成员 E：RoBERTa 与扩展实验

### 主模型

```text
M5 = RoBERTa-base
```

### 主技术路径

```text
roberta-base
→ three-class classification head
→ full fine-tuning on Combined
```

与 BERT 和 FinBERT 统一 train/val/test、max_length、调参预算、checkpoint metric 和 evaluator。

### 扩展实验优先级

第一优先级：

```text
RoBERTa-base on Combined
vs
RoBERTa-base on Combined-S
```

第二优先级，资源允许时二选一：

```text
RoBERTa-large
```

或：

```text
FLANG-RoBERTa-base
```

### 最终产出

```text
results/roberta_base_combined/
results/roberta_base_combined_split/
```

可选：

```text
results/roberta_large/
```

图片：

- RoBERTa vs BERT vs FinBERT；
- Combined vs Combined-S；
- RoBERTa confusion matrix；
- RoBERTa learning curve；
- 与 Shah paper result 的差异图。

### 文献职责

总结 Shah et al. (2023)。所有成员都至少阅读其 Dataset、Models、Results、FinBERT appendix 和 Annotation guide。

## 8. 预训练模型统一处理

### 可以作为主模型起点

| 类型 | 用法 |
|---|---|
| BERT-base | 在 FOMC 上 fine-tune |
| Finance-domain BERT | 在 FOMC 上 fine-tune |
| RoBERTa-base | 在 FOMC 上 fine-tune |
| GloVe | RNN embedding ablation |

### 不能当作小组自训模型

作者已经在 FOMC 数据上训练好的模型，例如：

```text
FOMC-RoBERTa
```

只能标记为：

```text
External Released Model
```

可以用于演示、sanity check、少量预测案例和外部参考，但不能放入主公平比较表并写成 `Our RoBERTa model`。

### 论文结果

放在 `Paper Reference` 列：

| Model | Our weighted F1 | Paper weighted F1 | Difference |
|---|---:|---:|---:|
| Rule-Based |  | 0.4966 |  |
| BiLSTM |  | 0.5387 |  |
| BERT-base |  | 0.6310 |  |
| FinBERT |  | 0.6325 |  |
| RoBERTa-base |  | 0.6755 |  |

论文分数是参考，不是必须完全复制的目标。

## 9. 统一评价指标

主指标：

```text
Weighted F1 mean ± standard deviation over 3 seeds
```

同时必须报告：

```text
Macro F1 mean ± std
Accuracy mean ± std
Dovish precision / recall / F1
Hawkish precision / recall / F1
Neutral precision / recall / F1
Training time
Inference time（可选）
Parameter count
```

## 10. 统一结果文件

每个模型目录：

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

`predictions.csv`：

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

`config.json` 至少记录：

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

## 11. 统一图片

所有成员使用成员 A 的 plotting script。

### Confusion Matrix

```text
Label order: Dovish, Hawkish, Neutral
Rows: True label
Columns: Predicted label
Normalisation: by true class
Display: percentage
```

### Learning Curve

仅训练模型需要：

```text
Training Loss
Validation Loss
Validation Weighted F1
Best Epoch marker
```

### Per-Class F1

固定顺序：

```text
Dovish
Hawkish
Neutral
```

## 12. 最终总图

成员 A 统一生成：

1. 所有模型 weighted F1，带 3 seeds 标准差；
2. 所有模型 macro F1；
3. 最佳传统 baseline、BiLSTM、BERT、FinBERT、RoBERTa 的 per-class F1；
4. Performance vs training time 或 parameter count；
5. 最佳 baseline 与最佳 Transformer confusion matrix；
6. RoBERTa Combined vs Combined-S。

## 13. 最终结果表

主表只放相同 `Combined unsplit` 数据：

| Model | Pretraining | Weighted F1 | Macro F1 | Accuracy | Dovish F1 | Hawkish F1 | Neutral F1 | Paper F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|

Ablation 表：

| Experiment | Variant A | Variant B | Difference |
|---|---|---|---:|
| Static pretraining | BiLSTM Random | BiLSTM GloVe |  |
| Fine-tuning | BERT Frozen | BERT Fine-Tuned |  |
| Domain pretraining | BERT | FinBERT |  |
| Task adaptation | FinBERT Zero-Shot | FinBERT Fine-Tuned |  |
| Preprocessing | RoBERTa Combined | RoBERTa Combined-S |  |
| Model scale（可选） | RoBERTa-base | RoBERTa-large |  |

## 14. 每名成员必须提交的文字

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

## 15. 执行顺序

### 阶段 1：协议和基础代码

成员 A 完成 data loader、splits、evaluator、result schema 和 plotting。  
成员 B–E 确认 checkpoint、建立代码骨架并完成论文摘要。

### 阶段 2：Baseline 和 Smoke Test

成员 A 跑完轻量 baseline。  
成员 B–E 各自在 seed 5768 上完成一次可运行模型。

### 阶段 3：统一调参

只使用：

```text
seed 5768 train + validation
```

### 阶段 4：冻结参数

冻结后运行：

```text
5768
78516
944601
```

### 阶段 5：统一汇总

成员 A 收集所有 `metrics.json` 和 `predictions.csv`，生成总表和总图。

### 阶段 6：扩展实验

主表完成后再做：

- BiLSTM + GloVe；
- Combined-S；
- RoBERTa-large 或 FLANG-RoBERTa；
- External released model demonstration。

## 16. 最终预期产出

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
15. RoBERTa-large 或 FLANG-RoBERTa
16. External FOMC-RoBERTa demonstration

> 五名成员负责五个工作包，最终形成约 12 个主结果和 2–4 个 ablation。所有实验共享相同数据、validation split、test discipline、评价指标、结果格式和绘图标准。预训练只在技术上适用的模型之间做受控比较，而不是所有成员重复运行全部 checkpoint。
