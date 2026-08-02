"""RoBERTa with hyperparameters selected by mean validation weighted-F1."""

from roberta_variant_runner import VariantConfig, run_variant


if __name__ == "__main__":
    run_variant(
        VariantConfig(
            name="roberta_average_weighted_hp",
            use_average_weighted_hp_search=True,
        )
    )
