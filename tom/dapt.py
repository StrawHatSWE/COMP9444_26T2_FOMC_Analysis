"""Step 2 - Domain-adaptive pretraining (masked-language-model continuation).

Continues MLM training of the encoder on unlabeled FOMC text before any task
fine-tuning. Default corpus: every training sentence across all variants/seeds,
filtered against every test sentence (TAPT). Pass --corpus for a larger
external Fed corpus (one sentence per line); it is filtered the same way.

Output checkpoint is a drop-in --model for search.py / train_final.py.

  python dapt.py                          # TAPT on task sentences, roberta-base
  python dapt.py --corpus fed_corpus.txt  # full DAPT
  python dapt.py --smoke                  # functionality check only
"""

import argparse
from pathlib import Path

from datasets import Dataset
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

import common


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="roberta-base")
    parser.add_argument("--corpus", type=Path, default=None,
                        help="optional text file, one sentence per line (filtered against test sentences)")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--mlm-probability", type=float, default=0.15)
    parser.add_argument("--output-dir", type=Path, default=common.RUNS_DIR / "dapt")
    parser.add_argument("--smoke", action="store_true", help="tiny corpus + a few steps, then exit")
    args = parser.parse_args()

    if args.corpus:
        raw = [l.strip() for l in args.corpus.read_text(encoding="utf-8").splitlines() if l.strip()]
        test_norm = common.all_test_sentences()
        sentences = [s for s in raw if common.norm_sentence(s) not in test_norm]
        print(f"corpus: {len(raw)} lines, {len(sentences)} after test-sentence filtering")
    else:
        sentences = common.unlabeled_corpus()
        print(f"TAPT corpus: {len(sentences)} unique train-side sentences (test-filtered)")

    if args.smoke:
        sentences = sentences[:32]
        args.epochs = 1.0
        args.output_dir = common.RUNS_DIR / "smoke-dapt"

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForMaskedLM.from_pretrained(args.model)

    ds = Dataset.from_dict({"text": sentences})
    ds = ds.map(lambda ex: tok(ex["text"], truncation=True, max_length=common.MAX_LENGTH),
                batched=True, remove_columns=["text"])

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(args.output_dir / "checkpoint"),
            learning_rate=args.lr,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            warmup_ratio=0.06,
            weight_decay=0.01,
            save_strategy="no",
            report_to=[],
            seed=common.SEARCH_SEED,
        ),
        train_dataset=ds,
        data_collator=DataCollatorForLanguageModeling(tok, mlm_probability=args.mlm_probability),
    )
    trainer.train()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tok.save_pretrained(args.output_dir)
    print(f"adapted encoder saved to {args.output_dir}")
    print(f"use it downstream:  python search.py --model {args.output_dir}")


if __name__ == "__main__":
    main()
