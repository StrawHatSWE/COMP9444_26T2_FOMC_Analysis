"""RoBERTa with label-smoothed cross-entropy."""

from roberta_variant_runner import VariantConfig, run_variant


if __name__ == "__main__":
    run_variant(
        VariantConfig(
            name="roberta_label_smoothing",
            label_smoothing=0.1,
        )
    )
