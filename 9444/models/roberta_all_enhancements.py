"""RoBERTa combining DAPT, averaged HP search, balancing, and smoothing."""

from roberta_variant_runner import VariantConfig, run_variant


if __name__ == "__main__":
    run_variant(
        VariantConfig(
            name="roberta_all_enhancements",
            use_dapt=True,
            use_average_weighted_hp_search=True,
            use_class_balancing=True,
            label_smoothing=0.1,
        )
    )
