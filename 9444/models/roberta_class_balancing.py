"""RoBERTa with inverse-frequency class-weighted cross-entropy."""

from roberta_variant_runner import VariantConfig, run_variant


if __name__ == "__main__":
    run_variant(
        VariantConfig(
            name="roberta_class_balancing",
            use_class_balancing=True,
        )
    )
