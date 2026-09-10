"""Tests for the experimental design generator.

Randomisation that is not tested is randomisation you are trusting. These check
the properties that actually matter for validity: the sequence is reproducible
from its seed, counterbalancing genuinely varies, combined blocks alternate
target and attribute trials, and no category repeats long enough for a
participant to stop reading.
"""

from __future__ import annotations

from collections import Counter

import pytest

from app.design import (
    DIMENSIONS,
    PRESETS,
    STUDIES,
    build_session,
    custom_study,
)


def test_seed_reproduces_the_exact_sequence():
    a = build_session(STUDIES["gin-premium"], preset="standard", seed=12345)
    b = build_session(STUDIES["gin-premium"], preset="standard", seed=12345)
    assert a["trials"] == b["trials"]
    assert a["counterbalance"] == b["counterbalance"]
    assert a["seed"] == b["seed"] == 12345


def test_different_seeds_give_different_sequences():
    a = build_session(STUDIES["gin-premium"], preset="standard", seed=1)
    b = build_session(STUDIES["gin-premium"], preset="standard", seed=2)
    assert a["trials"] != b["trials"]


def test_counterbalancing_actually_varies_across_seeds():
    """Both levels of the order factor must occur. A constant is not a design."""
    orders = {
        build_session(STUDIES["gin-premium"], seed=s)["counterbalance"]["congruent_pairing_first"]
        for s in range(40)
    }
    assert orders == {True, False}

    sides = {
        build_session(STUDIES["gin-premium"], seed=s)["counterbalance"]["target_a_starts_left"]
        for s in range(40)
    }
    assert sides == {True, False}


def test_seven_blocks_plus_calibration():
    s = build_session(STUDIES["gin-premium"], preset="full", seed=7)
    kinds = [b["kind"] for b in s["blocks"]]
    assert kinds[:2] == ["motor", "reading"]
    assert len([k for k in kinds if k not in ("motor", "reading")]) == 7
    task = [b for b in s["blocks"] if b["kind"] not in ("motor", "reading")]
    assert [b["index"] for b in task] == [1, 2, 3, 4, 5, 6, 7]


def test_block_five_has_thirty_trials_in_the_full_preset():
    """Greenwald et al. (2022) Appendix A raised Block 5 from 20 to 30."""
    assert PRESETS["full"]["counts"][5] == 30
    s = build_session(STUDIES["gin-premium"], preset="full", seed=3)
    b5 = next(b for b in s["blocks"] if b["index"] == 5)
    assert b5["n_trials"] == 30
    assert sum(1 for t in s["trials"] if t["block_index"] == 5) == 30


def test_only_combined_blocks_are_scored():
    s = build_session(STUDIES["gin-premium"], preset="standard", seed=11)
    scored_blocks = {b["index"] for b in s["blocks"] if b["role"]}
    assert scored_blocks == {3, 4, 6, 7}
    for t in s["trials"]:
        if t["block_index"] in (1, 2, 5) or t["block_index"] < 0:
            assert t["block_role"] is None
            assert t["pairing"] is None


def test_combined_blocks_alternate_target_and_attribute():
    """Without alternation the task decomposes into two separate sorting tasks."""
    s = build_session(STUDIES["gin-premium"], preset="full", seed=99)
    dim = STUDIES["gin-premium"].dimension
    attr_keys = {dim.pole_a.key, dim.pole_b.key}
    for block in (3, 4, 6, 7):
        seq = [t["stimulus_category"] for t in s["trials"] if t["block_index"] == block]
        is_attr = [c in attr_keys for c in seq]
        # Positions alternate: even indices are targets, odd are attributes.
        assert all(flag is (i % 2 == 1) for i, flag in enumerate(is_attr)), block


def test_no_category_runs_longer_than_three():
    s = build_session(STUDIES["gin-premium"], preset="full", seed=5)
    for block in (1, 2, 5):
        seq = [t["stimulus_category"] for t in s["trials"] if t["block_index"] == block]
        run = 1
        for i in range(1, len(seq)):
            run = run + 1 if seq[i] == seq[i - 1] else 1
            assert run <= 3, f"block {block} had a run of {run}"


def test_exemplars_are_evenly_exposed():
    """Sampling without replacement within a cycle keeps exposure even."""
    s = build_session(STUDIES["gin-premium"], preset="full", seed=21)
    targets = [t["stimulus"] for t in s["trials"]
               if t["stimulus_category"] == "target_a" and t["block_index"] > 0]
    counts = Counter(targets)
    assert max(counts.values()) - min(counts.values()) <= 1


def test_congruent_pairing_puts_target_a_with_pole_a():
    s = build_session(STUDIES["gin-premium"], preset="standard", seed=4)
    dim = STUDIES["gin-premium"].dimension
    for block in s["blocks"]:
        if block["kind"] != "combined":
            continue
        left = {c["key"] for c in block["left"]}
        right = {c["key"] for c in block["right"]}
        together = (
            ("target_a" in left and dim.pole_a.key in left)
            or ("target_a" in right and dim.pole_a.key in right)
        )
        assert together is (block["pairing"] == "congruent"), block["index"]


def test_key_mapping_matches_the_side_the_category_is_on():
    s = build_session(STUDIES["gin-premium"], preset="standard", seed=8)
    by_index = {b["index"]: b for b in s["blocks"]}
    for t in s["trials"]:
        block = by_index[t["block_index"]]
        if block["kind"] in ("motor", "reading"):
            continue
        left = {c["key"] for c in block["left"]}
        expected = "E" if t["stimulus_category"] in left else "I"
        assert t["correct_key"] == expected


def test_reversed_block_swaps_the_targets():
    s = build_session(STUDIES["gin-premium"], preset="standard", seed=6)
    b1 = next(b for b in s["blocks"] if b["index"] == 1)
    b5 = next(b for b in s["blocks"] if b["index"] == 5)
    assert [c["key"] for c in b1["left"]] != [c["key"] for c in b5["left"]]


# --------------------------------------------------------------------------
# SC-IAT
# --------------------------------------------------------------------------


def test_sciat_has_four_blocks_and_only_two_are_scored():
    s = build_session(STUDIES["aurelia-sciat"], preset="standard", seed=2)
    assert s["instrument"] == "sciat"
    task = [b for b in s["blocks"] if b["kind"] == "sciat"]
    assert len(task) == 4
    assert [b["role"] for b in task] == [None, "sciat", None, "sciat"]


def test_sciat_uses_the_seven_seven_ten_ratio():
    """Without the ratio correction one key is correct more often than the other."""
    s = build_session(STUDIES["aurelia-sciat"], preset="full", seed=13)
    for block in s["blocks"]:
        if block["kind"] != "sciat":
            continue
        seq = [t["stimulus_category"] for t in s["trials"] if t["block_index"] == block["index"]]
        counts = Counter(seq)
        total = len(seq)
        target = counts["target_a"] / total
        # 7 / 24 = 0.2917. Allow for the cycle not dividing evenly.
        assert target == pytest.approx(7 / 24, abs=0.06), (block["index"], counts)


def test_sciat_response_window_is_set_on_critical_blocks():
    s = build_session(STUDIES["aurelia-sciat"], preset="standard", seed=14)
    for b in s["blocks"]:
        if b["kind"] == "sciat":
            assert b["response_window_ms"] == 1500


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------


def test_calibration_blocks_produce_usable_trials():
    s = build_session(STUDIES["gin-premium"], preset="full", seed=17)
    motor = [t for t in s["trials"] if t["stimulus_category"].startswith("motor")]
    reading = [t for t in s["trials"] if t["stimulus_category"] == "reading"]
    assert len(motor) == PRESETS["full"]["calibration"]["motor"]
    assert len(reading) == PRESETS["full"]["calibration"]["reading"]
    # The reading block must span more than one length or no slope is estimable.
    assert len({t["word_count"] for t in reading}) >= 2
    assert all(t["correct_key"] in ("E", "I") for t in motor)


def test_calibration_can_be_switched_off():
    s = build_session(STUDIES["gin-premium"], seed=1, include_calibration=False)
    assert all(b["kind"] not in ("motor", "reading") for b in s["blocks"])


# --------------------------------------------------------------------------
# Custom studies
# --------------------------------------------------------------------------


def test_custom_study_builds_and_is_flagged_as_not_fictitious():
    st = custom_study("Brand X", "Brand Y", dimension_key="trust")
    assert st.instrument == "iat"
    assert st.fictitious is False
    assert st.dimension.key == "trust"
    s = build_session(st, preset="express", seed=1)
    assert s["n_trials"] > 0


def test_custom_study_without_a_comparator_is_a_single_category_test():
    st = custom_study("Brand X", None)
    assert st.instrument == "sciat"


def test_every_registered_study_generates_at_every_preset():
    for study in STUDIES.values():
        for preset in PRESETS:
            s = build_session(study, preset=preset, seed=1)
            assert s["n_trials"] > 0
            assert s["n_scored_trials"] > 0
            assert all(t["correct_key"] in ("E", "I") for t in s["trials"])


def test_dimensions_are_well_formed():
    for dim in DIMENSIONS.values():
        assert dim.pole_a.key == "positive"
        assert dim.pole_b.key == "negative"
        assert len(dim.pole_a.stimuli) >= 5
        assert len(dim.pole_b.stimuli) >= 5
        assert not set(dim.pole_a.stimuli) & set(dim.pole_b.stimuli)
