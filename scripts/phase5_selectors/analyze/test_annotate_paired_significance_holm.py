from scripts.phase5_selectors.analyze.annotate_paired_significance_holm import (
    annotate,
    holm_adjust,
)


def test_holm_adjust_is_monotone_in_sorted_order() -> None:
    adjusted = holm_adjust([("a", 0.01), ("b", 0.03), ("c", 0.2)])
    assert adjusted == {"a": 0.03, "b": 0.06, "c": 0.2}


def test_annotate_separates_primary_secondary_and_diagnostic() -> None:
    names = ("primary", "secondary_a", "secondary_b", "diagnostic")
    payload = {
        "comparisons": [
            {
                "name": name,
                "paired_randomization": {
                    "macro_f1": {"p_value_two_sided": value}
                },
            }
            for name, value in zip(names, (0.04, 0.01, 0.04, 0.001))
        ]
    }
    result = annotate(
        payload,
        primary="primary",
        secondary=("secondary_a", "secondary_b"),
        diagnostic=("diagnostic",),
        metric="macro_f1",
        alpha=0.05,
    )
    block = result["multiple_testing"]
    assert block["holm_adjusted_p_values"] == {
        "secondary_a": 0.02,
        "secondary_b": 0.04,
    }
    assert block["diagnostic_comparisons_excluded_from_family"] == ["diagnostic"]
