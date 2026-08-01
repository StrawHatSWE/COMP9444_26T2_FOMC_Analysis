"""RoBERTa with domain-adaptive pretraining (DAPT) on FOMC training text."""

from roberta_variant_runner import VariantConfig, run_variant


if __name__ == "__main__":
    run_variant(
        VariantConfig(
            name="roberta_dapt",
            use_dapt=True,
        )
    )
