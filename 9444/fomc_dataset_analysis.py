"""
Basic dataset analysis for COMP9444 project 003:
NLP Classification of FOMC Texts for Market Analysis.

The script uses only Python standard-library XML/zip readers so it does not
depend on pandas/openpyxl.
"""

from collections import Counter
from io import BytesIO
from pathlib import Path
import statistics
from zipfile import ZipFile
import xml.etree.ElementTree as ET


DATA_ZIP = Path(__file__).with_name("FOMC_dataset_checkpoint.zip")
NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
LABEL_NAMES = {0: "Dovish", 1: "Hawkish", 2: "Neutral"}


def column_index(cell_ref):
    letters = "".join(ch for ch in cell_ref if ch.isalpha())
    idx = 0
    for ch in letters:
        idx = idx * 26 + ord(ch) - ord("A") + 1
    return idx - 1


def cell_text(cell, shared_strings):
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        return "".join(t.text or "" for t in cell.iter(NS + "t"))

    value = cell.find(NS + "v")
    if value is None:
        return ""
    if cell_type == "s":
        return shared_strings[int(value.text)]
    return value.text or ""


def read_xlsx_from_zip(data):
    with ZipFile(BytesIO(data)) as workbook:
        shared_strings = []
        if "xl/sharedStrings.xml" in workbook.namelist():
            root = ET.fromstring(workbook.read("xl/sharedStrings.xml"))
            for item in root:
                shared_strings.append("".join(t.text or "" for t in item.iter(NS + "t")))

        sheet = ET.fromstring(workbook.read("xl/worksheets/sheet1.xml"))
        rows = []
        for row in sheet.find(NS + "sheetData"):
            values = []
            for cell in row:
                idx = column_index(cell.attrib["r"])
                while len(values) <= idx:
                    values.append("")
                values[idx] = cell_text(cell, shared_strings)
            rows.append(values)

        headers = rows[0]
        records = []
        for values in rows[1:]:
            record = {headers[i]: values[i] if i < len(values) else "" for i in range(len(headers))}
            record["index"] = int(record["index"])
            record["year"] = int(record["year"])
            record["label"] = int(record["label"])
            records.append(record)
        return records


def load_seed(seed):
    records = []
    with ZipFile(DATA_ZIP) as archive:
        for split in ("train", "test"):
            name = f"FOMC_dataset_checkpoint/lab-manual-combine-{split}-{seed}.xlsx"
            split_records = read_xlsx_from_zip(archive.read(name))
            for record in split_records:
                record["seed"] = seed
                record["split"] = split
            records.extend(split_records)
    return records


def summarize(records):
    labels = Counter(record["label"] for record in records)
    years = [record["year"] for record in records]
    sentence_lengths = [len(record["sentence"].split()) for record in records]
    char_lengths = [len(record["sentence"]) for record in records]
    split_counts = Counter(record["split"] for record in records)
    duplicate_sentence_count = len(records) - len({record["sentence"] for record in records})

    return {
        "total": len(records),
        "split_counts": split_counts,
        "labels": labels,
        "year_min": min(years),
        "year_max": max(years),
        "top_years": Counter(years).most_common(5),
        "word_mean": statistics.mean(sentence_lengths),
        "word_median": statistics.median(sentence_lengths),
        "word_max": max(sentence_lengths),
        "char_mean": statistics.mean(char_lengths),
        "char_median": statistics.median(char_lengths),
        "char_max": max(char_lengths),
        "empty_sentences": sum(1 for record in records if not record["sentence"].strip()),
        "duplicate_sentence_count": duplicate_sentence_count,
    }


def print_summary(seed, summary):
    print(f"\nSeed {seed}")
    print("-" * 60)
    print(f"Total labelled samples: {summary['total']}")
    print(f"Train/test split: {summary['split_counts']['train']} / {summary['split_counts']['test']}")
    print("Class distribution:")
    for label in (0, 1, 2):
        count = summary["labels"][label]
        pct = 100 * count / summary["total"]
        print(f"  {label} {LABEL_NAMES[label]:7s}: {count:4d} ({pct:5.1f}%)")
    print(f"Year range: {summary['year_min']} - {summary['year_max']}")
    print(f"Top years by sample count: {summary['top_years']}")
    print(
        "Sentence length in words: "
        f"mean={summary['word_mean']:.2f}, "
        f"median={summary['word_median']:.0f}, "
        f"max={summary['word_max']}"
    )
    print(
        "Sentence length in characters: "
        f"mean={summary['char_mean']:.2f}, "
        f"median={summary['char_median']:.0f}, "
        f"max={summary['char_max']}"
    )
    print(f"Empty sentences: {summary['empty_sentences']}")
    print(f"Repeated sentence texts: {summary['duplicate_sentence_count']}")


def main():
    for seed in ("5768", "78516", "944601"):
        records = load_seed(seed)
        print_summary(seed, summarize(records))


if __name__ == "__main__":
    main()
