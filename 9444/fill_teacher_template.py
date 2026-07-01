"""
Fill the official COMP9444 presentation template with Week 5 FOMC checkpoint
content, without requiring python-pptx.
"""

from copy import deepcopy
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
import re
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "COMP9444_Presentation_TeamName.pptx"
OUTPUT = Path(__file__).with_name("COMP9444_Presentation_FOMC_Week5_TemplateFilled.pptx")

NS = {
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
}
for prefix, uri in NS.items():
    ET.register_namespace(prefix, uri)


SLIDES = {
    1: {
        "title": "NLP Classification of FOMC Texts for Market Analysis",
        "body": (
            "Team Name / zIDs: [fill in before submission]\n"
            "Week 5 Checkpoint: Introduction, Literature Review, and Dataset Analysis\n"
            "Project 003\n"
            "Task: classify FOMC sentences as Dovish, Hawkish, or Neutral"
        ),
        "presenter": "Presenter: Team Name / zIDs",
    },
    2: {
        "title": "Motivation",
        "body": (
            "Why FOMC text matters\n"
            "• FOMC statements, meeting minutes, press conferences and speeches influence expectations about interest rates, inflation, bonds and equity markets.\n"
            "• A single FOMC communication can trigger large market reactions, so extracting policy stance is practically valuable.\n"
            "• Manual reading is slow, subjective and hard to scale across decades of documents.\n"
            "• Standard positive/negative sentiment does not directly capture monetary-policy stance.\n"
            "• NLP can convert raw FOMC text into measurable stance signals for later market analysis."
        ),
        "presenter": "Presenter: [fill in]",
    },
    3: {
        "title": "Problem Statement",
        "body": (
            "Sentence-level policy stance classification\n"
            "• Input: one sentence from FOMC-related communication.\n"
            "• Output classes: 0 Dovish, 1 Hawkish, 2 Neutral.\n"
            "• Dovish: implies easing, accommodation, lower-rate pressure or support for employment/growth.\n"
            "• Hawkish: implies tightening, inflation control, higher-rate pressure or reduced accommodation.\n"
            "• Neutral: factual, mixed, or not directly related to monetary-policy stance.\n"
            "• Evaluation should include macro-F1 and per-class F1 because Neutral is the largest class."
        ),
        "presenter": "Presenter: [fill in]",
    },
    4: {
        "title": "Literature Review",
        "body": (
            "Shah et al. (2023): Trillion Dollar Words\n"
            "• Introduces a large FOMC corpus and a hawkish-dovish-neutral sentence classification task.\n"
            "• Collects meeting minutes, press conference transcripts and speeches from 1996-2022.\n"
            "• Shows ordinary financial sentiment is insufficient: words like 'increase' depend on the economic object.\n"
            "• Benchmarks rule-based methods, LSTM, BERT-family models, finance-domain models and RoBERTa-large.\n"
            "• Uses the best stance model to construct a monetary-policy stance measure for market analysis.\n"
            "• This paper is directly relevant because it defines our dataset, task and strong baseline context."
        ),
        "presenter": "Presenter: [fill in]",
    },
    5: {
        "title": "Literature Review",
        "body": (
            "Method context and baselines\n"
            "• Rule-based dictionaries are interpretable but brittle for subtle policy language.\n"
            "• FinBERT-style models capture finance sentiment, but sentiment is not the same as hawkish/dovish stance.\n"
            "• Transformer classifiers can learn domain-sensitive sentence representations after fine-tuning.\n"
            "• For our project, start with reproducible baselines before advanced models.\n"
            "• Proposed benchmarks: majority class, TF-IDF + Logistic Regression/Linear SVM, and BERT/FinBERT/RoBERTa.\n"
            "• Clearly separate our own implementation from any released FOMC-RoBERTa model."
        ),
        "presenter": "Presenter: [fill in]",
    },
    6: {
        "title": "Dataset(s)",
        "body": (
            "FOMC labelled train/test splits\n"
            "• Source: gtfintechlab/fomc-hawkish-dovish; paper: Shah et al. (2023).\n"
            "• Sources include meeting minutes, press conference transcripts and Federal Reserve speeches.\n"
            "• Raw corpus spans 1996-2022; press conferences begin in 2011.\n"
            "• Provided checkpoint format: Excel .xlsx files.\n"
            "• Columns: index, sentence, year, label.\n"
            "• Three random seeds: 5768, 78516, 944601.\n"
            "• Each seed has 1,903 train samples and 476 test samples."
        ),
        "presenter": "Presenter: [fill in]",
    },
    7: {
        "title": "Data Analysis",
        "body": (
            "Basic class distribution\n"
            "• Total labelled samples per seed: 2,379.\n"
            "• Train/test split: 1,903 / 476.\n"
            "• Label 0 Dovish: 625 samples (26.3%).\n"
            "• Label 1 Hawkish: 600 samples (25.2%).\n"
            "• Label 2 Neutral: 1,154 samples (48.5%).\n"
            "• Neutral is the largest class, so accuracy alone can be misleading.\n"
            "• Recommended metrics: macro-F1, weighted-F1, per-class F1 and confusion matrix."
        ),
        "presenter": "Presenter: [fill in]",
    },
    8: {
        "title": "Data Analysis",
        "body": (
            "Coverage, text length and quality\n"
            "• Year range: 1996-2022, covering different monetary-policy regimes.\n"
            "• Mean sentence length: 31.54 words.\n"
            "• Median sentence length: 29 words.\n"
            "• Maximum sentence length: 181 words.\n"
            "• Empty sentences: 0.\n"
            "• Repeated sentence texts: 53.\n"
            "• Long technical sentences mean transformer truncation and tokenization settings should be checked."
        ),
        "presenter": "Presenter: [fill in]",
    },
    9: {
        "title": "Discussion",
        "body": (
            "Dataset quality and modelling challenges\n"
            "• Domain specificity: the same word can imply different policy stances depending on context.\n"
            "• Sentence context: some sentences mix cautious language and policy signals.\n"
            "• Class imbalance: the model may over-predict Neutral if not monitored.\n"
            "• Temporal shift: Fed communication style changes across 1996-2022.\n"
            "• Repeated sentences should be checked for leakage or overfitting risk.\n"
            "• Final reporting should specify which random seed(s) and split(s) were used."
        ),
        "presenter": "Presenter: [fill in]",
    },
    10: {
        "title": "Conclusion",
        "body": (
            "Checkpoint summary and next steps\n"
            "• Read Shah et al. (2023) and confirmed it is directly relevant to our project.\n"
            "• Drafted motivation and problem statement for FOMC stance classification.\n"
            "• Inspected the dataset format, label mapping, class balance, year coverage and text length.\n"
            "• Next: implement reproducible loaders for all three seeds.\n"
            "• Next: run majority-class and TF-IDF + Logistic Regression baselines.\n"
            "• Later: compare BERT/FinBERT/RoBERTa-style models and perform error analysis."
        ),
        "presenter": "Presenter: [fill in]",
    },
}


def shape_text(shape):
    texts = []
    for node in shape.iter(f"{{{NS['a']}}}t"):
        if node.text:
            texts.append(node.text)
    return "".join(texts)


def text_shapes(root):
    shapes = []
    for shape in root.iter(f"{{{NS['p']}}}sp"):
        if shape.find("p:txBody", NS) is not None and shape_text(shape).strip():
            shapes.append(shape)
    return shapes


def set_text(shape, text):
    tx_body = shape.find("p:txBody", NS)
    if tx_body is None:
        return

    existing_rpr = None
    first_run = tx_body.find(".//a:r", NS)
    if first_run is not None:
        rpr = first_run.find("a:rPr", NS)
        if rpr is not None:
            existing_rpr = deepcopy(rpr)

    for child in list(tx_body):
        if child.tag == f"{{{NS['a']}}}p":
            tx_body.remove(child)

    for line in text.split("\n"):
        p = ET.SubElement(tx_body, f"{{{NS['a']}}}p")
        r = ET.SubElement(p, f"{{{NS['a']}}}r")
        if existing_rpr is not None:
            r.append(deepcopy(existing_rpr))
        t = ET.SubElement(r, f"{{{NS['a']}}}t")
        t.text = line


def update_slide(xml_data, slide_no):
    root = ET.fromstring(xml_data)
    shapes = text_shapes(root)
    content = SLIDES.get(slide_no)
    if not content:
        return xml_data

    if len(shapes) >= 1:
        set_text(shapes[0], content["title"])
    if len(shapes) >= 2:
        set_text(shapes[1], content["body"])
    if len(shapes) >= 4:
        set_text(shapes[-1], content["presenter"])
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def main():
    slide_re = re.compile(r"ppt/slides/slide(\d+)\.xml$")
    with ZipFile(TEMPLATE, "r") as src, ZipFile(OUTPUT, "w", ZIP_DEFLATED) as dst:
        for item in src.infolist():
            data = src.read(item.filename)
            match = slide_re.match(item.filename)
            if match:
                slide_no = int(match.group(1))
                data = update_slide(data, slide_no)
            dst.writestr(item, data)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
