from __future__ import annotations

from tools.run_v16_externality_conditioned_feedback import (
    CONTROL_ROSTER,
    EXTERNALITY_CONTROLS,
)


def test_v16_control_roster_separates_primary_mechanism_and_ablations() -> None:
    ids = [row["control_id"] for row in CONTROL_ROSTER]
    assert len(ids) == len(set(ids)) == 6
    assert ids == [
        "NO_FEEDBACK",
        "DELIVERY_RATIO_CF_BETA_050",
        "EXTERNALITY_CF_BETA_025",
        "EXTERNALITY_CF_BETA_050",
        "EXTERNALITY_CF_BETA_100",
        "UNIFORM_MATCHED_EXTERNALITY_CF_BETA_050",
    ]
    assert set(EXTERNALITY_CONTROLS) == {
        "EXTERNALITY_CF_BETA_025",
        "EXTERNALITY_CF_BETA_050",
        "EXTERNALITY_CF_BETA_100",
    }


def test_v16_beta_grid_is_predeclared_and_not_single_point() -> None:
    beta = {
        float(row["beta"])
        for row in CONTROL_ROSTER
        if row["control_id"] in EXTERNALITY_CONTROLS
    }
    assert beta == {0.25, 0.5, 1.0}


def test_delivery_ratio_filter_is_explicitly_an_ablation() -> None:
    row = next(
        item for item in CONTROL_ROSTER
        if item["control_id"] == "DELIVERY_RATIO_CF_BETA_050"
    )
    assert row["control_role"] == "DELIVERY_ONLY_RATIO_FILTER_ABLATION"
