# Week 5 Checkpoint 中文说明

项目：**003. NLP Classification of FOMC Texts for Market Analysis**

本周 checkpoint 要求完成三件事：

1. 在老师给的 presentation template 中草拟 **Motivation / Problem Statement**。
2. 至少阅读一篇相关论文，并在 slide deck 中写 **Literature Review**。
3. 对数据集做基础分析，包括格式、样本数、类别数、类别分布和数据质量。

我已经整理出：

- 模板填充版 PPT：`COMP9444_Presentation_FOMC_Week5_TemplateFilled.pptx`
- 原 FOMC 草稿版 PPT：`COMP9444_FOMC_Checkpoint_Completed.pptx`
- 中文说明：`week5_checkpoint_summary_zh.md`
- 英文摘要说明：`week5_checkpoint_summary.md`
- 数据分析脚本：`fomc_dataset_analysis.py`

---

## 1. 项目任务说明

项目名称：**NLP Classification of FOMC Texts for Market Analysis**

这个项目的核心任务是：给定一句来自 FOMC 相关材料的英文句子，训练 NLP 模型判断它表达的货币政策立场。

输入：

```text
一条 FOMC 相关英文句子
```

输出：

```text
0 = Dovish  鸽派：倾向宽松政策、降息、支持就业或经济刺激
1 = Hawkish 鹰派：倾向紧缩政策、控通胀、加息或减少宽松
2 = Neutral 中性：事实陈述、混合信号，或没有明显政策立场
```

为什么这个任务重要：

- FOMC 声明、会议纪要、新闻发布会和官员讲话会影响市场对利率、通胀、债券和股票的预期。
- 人工阅读 FOMC 文本速度慢、主观性强，难以扩展到几十年的文本。
- 普通金融情绪分析只判断 positive / negative，但货币政策文本需要判断 hawkish / dovish。比如 “increase” 如果指 employment 可能偏鸽派，如果指 inflation 或 rates 可能偏鹰派。
- 如果能把文本转成结构化 stance label，就可以进一步用于 market analysis，例如观察政策措辞变化与 Treasury yields、stock market 或 CPI/PPI 的关系。

本 checkpoint 阶段的目标不是完成最终模型，而是先完成：

1. 明确 motivation 和 problem statement。
2. 阅读并总结相关论文。
3. 对数据集做基础探索，理解样本格式、类别分布和潜在建模难点。

---

## 2. 论文简介

使用论文：

**Shah, Paturi, and Chava (2023), "Trillion Dollar Words: A New Financial Dataset, Task & Market Analysis"**

这篇论文研究的问题是：FOMC 公开沟通文本如何影响金融市场，以及能否用 NLP 模型自动提取其中的货币政策立场。

论文的主要贡献：

1. **提出新任务。** 论文将 FOMC 句子分类为 hawkish、dovish 或 neutral，而不是传统的 positive / negative sentiment。
2. **构建数据集。** 作者收集并标注了 FOMC meeting minutes、press conferences 和 speeches，覆盖 1996-2022 年。
3. **比较多种模型。** 论文测试了 rule-based 方法、LSTM、BERT-family、金融领域模型和 RoBERTa-large 等模型。
4. **提出政策立场指标。** 作者用分类模型构造 aggregate monetary policy stance measure。
5. **连接市场分析。** 论文进一步分析文本立场指标与 Treasury market、stock market、CPI、PPI 等经济金融变量的关系。

对我们项目的启发：

- 这篇论文说明普通金融情绪模型不够，因为 FOMC 文本中的政策含义高度依赖上下文。
- 它提供了数据来源、标签定义和强 baseline，是我们项目最直接的参考。
- 它也提醒我们最终不能只看 accuracy，还要看 macro-F1、per-class F1 和错误类型。

---

## 3. 论文是否合理？

合理，而且很适合作为本项目 checkpoint 的核心论文。

使用论文：

**Shah, Paturi, and Chava (2023), "Trillion Dollar Words: A New Financial Dataset, Task & Market Analysis"**

原因：

1. 这篇论文就是 FOMC hawkish / dovish / neutral 分类任务和数据集的来源论文。
2. 论文定义的任务与我们的项目完全一致：对 FOMC 相关文本的句子进行货币政策立场分类。
3. 论文解释了为什么普通 financial sentiment analysis 不够用：正面/负面情绪不等于 hawkish/dovish 政策立场。
4. 论文提供了多个 baseline 和模型方向，包括 rule-based、LSTM、BERT-family、finance-domain models 和 RoBERTa-large。
5. 论文不仅做分类，还把文本立场指标与 Treasury yields、stock market、CPI/PPI 等市场变量联系起来，和我们的 “Market Analysis” 项目目标一致。

需要注意的限制：

- 这篇论文和数据集太接近，适合作为主论文，但最终 presentation/report 最好再补 2-4 篇相关论文。
- 论文中最强模型是大型预训练模型；我们项目初期应该先做可复现的简单 baseline。
- 市场影响分析比文本分类更复杂，不能把分类准确率直接等同于市场预测能力。

结论：

**Shah et al. 2023 可以作为 Week 5 checkpoint 的主 literature review 论文；后续 Week 10 前建议补 FinBERT、Loughran-McDonald、Hansen-McMahon 等相关工作。**

---

## 4. PPT 内容安排

模板填充版 PPT 覆盖了 Week 5 checkpoint 要求：

| 页码 | 内容 |
|---:|---|
| 1 | 项目标题与团队信息占位 |
| 2 | Motivation：为什么 FOMC 文本重要 |
| 3 | Problem Statement：句子级 hawkish/dovish/neutral 分类 |
| 4 | Literature Review：Shah et al. 2023 核心贡献 |
| 5 | Literature Review：方法背景与 baseline |
| 6 | Dataset：数据来源、格式、标签含义 |
| 7 | Data Analysis：类别分布 |
| 8 | Data Analysis：年份覆盖与文本长度 |
| 9 | Discussion：数据质量与建模挑战 |
| 10 | Conclusion / Next Steps：下一步计划 |

---

## 5. 数据集格式

数据包：

```text
FOMC_dataset_checkpoint.zip
```

包含 3 个随机种子的 train/test split：

```text
lab-manual-combine-train-5768.xlsx
lab-manual-combine-test-5768.xlsx
lab-manual-combine-train-78516.xlsx
lab-manual-combine-test-78516.xlsx
lab-manual-combine-train-944601.xlsx
lab-manual-combine-test-944601.xlsx
```

每个 Excel 文件有 4 列：

| 列名 | 含义 |
|---|---|
| index | 原始样本编号 |
| sentence | FOMC 相关文本句子 |
| year | 文本对应年份 |
| label | 类别标签 |

标签含义：

```text
0 = Dovish
1 = Hawkish
2 = Neutral
```

---

## 6. 基础数据分析结果

三个 seed 的总样本池一致，只是 train/test 切分不同。

每个 seed：

| 指标 | 数值 |
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

解释：

- 这是一个 **3-class sentence classification** 任务。
- Neutral 是最大类别，占 48.5%，存在中等程度类别不平衡。
- Dovish 和 Hawkish 数量接近，分别约 26.3% 和 25.2%。
- 因为 Neutral 最大，不能只看 accuracy，建议报告 macro-F1、weighted-F1、per-class F1 和 confusion matrix。
- 文本跨越 1996-2022，不同年份可能存在 monetary-policy regime 和 communication style 的变化。
- 句子平均约 31.54 个词，最长 181 个词，后续使用 transformer 时要注意 truncation。

---

## 7. 数据质量与建模挑战

主要挑战：

1. **领域语言强。** 例如 “increase” 可能指 employment、inflation、rates 或 accommodation，不同对象对应不同政策含义。
2. **句子上下文依赖。** 单句可能同时包含谨慎措辞和政策信号，Neutral 与 stance 类别边界不总是清晰。
3. **类别不平衡。** Neutral 接近一半，模型可能倾向预测 Neutral。
4. **时间漂移。** 1996-2022 期间 Fed communication style 和宏观经济环境变化很大。
5. **重复句子。** 有 53 条 repeated sentence texts，后续要确认是否会造成数据泄漏或过拟合风险。

---

## 8. 建议 baseline / benchmark

项目描述里提到需要 3 个 benchmark。建议：

1. **Majority-class baseline**：永远预测 Neutral。
2. **Traditional ML baseline**：TF-IDF + Logistic Regression 或 Linear SVM。
3. **Neural / Transformer baseline**：BERT、FinBERT、RoBERTa，或如果允许，比较 released FOMC-RoBERTa。

建议指标：

- Accuracy
- Macro-F1
- Weighted-F1
- Per-class precision / recall / F1
- Confusion matrix

---

## 9. 如何复现数据分析

运行：

```bash
cd /Users/lanxiaoyu/Desktop/9444
/Users/lanxiaoyu/miniconda3/envs/unsw/bin/python3 group/tutweek5/fomc_dataset_analysis.py
```

脚本只使用 Python 标准库读取 `.xlsx` 的 zip/xml 结构，不依赖 pandas 或 openpyxl。

---

## 10. 需要手动补的信息

PPT 第 1 页还需要你们自己填：

```text
Team Name / zIDs
```

如果 tutor 要求每页 presenter，也可以把 footer 里的 placeholder presenter 改成具体组员姓名和 zID。
