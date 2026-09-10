"""Experimental design: stimulus sets, block structure and counterbalancing.

The design is built **on the server** and sent to the browser as a fully
enumerated trial list. That is a deliberate choice with three consequences:

1.  *Reproducibility.* A session is a ``(study_id, seed)`` pair. Re-running with
    the same seed regenerates a byte-identical trial sequence, so an analysis
    can be replayed months later against the exact stimuli the participant saw.
2.  *Auditability.* Randomisation and counterbalancing are recorded facts, not
    something that happened in a browser and was forgotten.
3.  *Blindness.* The client never decides which condition it is in, so a
    front-end bug cannot silently unbalance the design.

Block structure follows the seven-block IAT of Greenwald, Nosek & Banaji (2003),
with one update from Greenwald, Brendl et al. (2022, Appendix A): Block 5, the
reversed-target discrimination, runs 30 trials rather than the original 20,
because "an increase from 20 to 30 trials in Block 5 was adopted as a procedure
that often keeps the effect of order of combined tasks to a minimum". The 20/40
practice/test asymmetry that remains is, in the same paper's phrase, a
"historical accident" rather than a designed feature.

Two calibration blocks run before the task proper. Neither is scored into D:

*   **Motor calibration** — press the cued key, 20 trials. Gives a baseline
    movement time for this participant on this device, which is what separates
    "slow because the association is weak" from "slow because they are using a
    trackpad on a train".
*   **Reading calibration** — read a phrase and press to continue, 10 trials
    across a range of word counts. Gives a per-participant reading slope, so a
    longer attribute label is not mistaken for a weaker association.

This two-part calibration is what commercial implicit-testing platforms add on
top of the academic instrument, and it is reported here as a covariate rather
than used to adjust D (D already standardises within-person).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

Side = Literal["left", "right"]

LEFT_KEY = "E"
RIGHT_KEY = "I"


# --------------------------------------------------------------------------
# Attribute dimensions
# --------------------------------------------------------------------------

# The canonical evaluative words used across the IAT literature. Kept unchanged
# so that a D-score from this tool sits on the same scale as published work;
# swapping in bespoke adjectives would make the numbers incomparable to
# anything.
POSITIVE_WORDS = [
    "Joy", "Love", "Peace", "Wonderful", "Pleasure",
    "Glorious", "Laughter", "Happy", "Delight", "Excellent",
]
NEGATIVE_WORDS = [
    "Agony", "Terrible", "Horrible", "Nasty", "Evil",
    "Awful", "Failure", "Hurt", "Painful", "Dreadful",
]

# A commercial attribute dimension. Valence answers "do they like it"; a brand
# manager usually needs "what does it mean to them", and premium-vs-everyday is
# the dimension most often behind a pack, price or positioning decision.
PREMIUM_WORDS = [
    "Crafted", "Refined", "Exclusive", "Luxurious", "Sophisticated",
    "Artisan", "Heritage", "Considered",
]
EVERYDAY_WORDS = [
    "Ordinary", "Basic", "Routine", "Generic", "Cheap",
    "Mass-market", "Functional", "Plain",
]

TRUST_WORDS = ["Reliable", "Honest", "Safe", "Dependable", "Straightforward", "Solid"]
RISK_WORDS = ["Unreliable", "Shady", "Risky", "Erratic", "Confusing", "Flimsy"]


@dataclass(slots=True)
class Category:
    """A category label plus the stimuli that belong to it."""

    key: str
    label: str
    stimuli: list[str]
    colour: str = "#f4f4f5"

    def to_dict(self) -> dict:
        return {"key": self.key, "label": self.label, "stimuli": self.stimuli, "colour": self.colour}


ATTRIBUTE_COLOUR = "#4ade80"
TARGET_COLOUR = "#a78bfa"
TARGET_COLOUR_B = "#38bdf8"


@dataclass(slots=True)
class AttributeDimension:
    """The two poles a target can be associated with.

    ``pole_a`` is the pole that defines the "congruent" pairing. That is a
    labelling convention only — it encodes no expectation about which way the
    result will go.
    """

    key: str
    name: str
    pole_a: Category
    pole_b: Category

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "name": self.name,
            "pole_a": self.pole_a.to_dict(),
            "pole_b": self.pole_b.to_dict(),
        }


DIMENSIONS: dict[str, AttributeDimension] = {
    "valence": AttributeDimension(
        key="valence", name="Positive vs Negative",
        pole_a=Category("positive", "Positive", POSITIVE_WORDS, ATTRIBUTE_COLOUR),
        pole_b=Category("negative", "Negative", NEGATIVE_WORDS, ATTRIBUTE_COLOUR),
    ),
    "premium": AttributeDimension(
        key="premium", name="Premium vs Everyday",
        pole_a=Category("positive", "Premium", PREMIUM_WORDS, ATTRIBUTE_COLOUR),
        pole_b=Category("negative", "Everyday", EVERYDAY_WORDS, ATTRIBUTE_COLOUR),
    ),
    "trust": AttributeDimension(
        key="trust", name="Trustworthy vs Risky",
        pole_a=Category("positive", "Trustworthy", TRUST_WORDS, ATTRIBUTE_COLOUR),
        pole_b=Category("negative", "Risky", RISK_WORDS, ATTRIBUTE_COLOUR),
    ),
}


# --------------------------------------------------------------------------
# Studies
# --------------------------------------------------------------------------


@dataclass(slots=True)
class Study:
    """A complete study definition."""

    id: str
    name: str
    sector: str
    blurb: str
    target_a: Category
    target_b: Category | None
    dimension: AttributeDimension
    fictitious: bool = True

    @property
    def instrument(self) -> str:
        return "iat" if self.target_b else "sciat"

    @property
    def positive(self) -> Category:
        return self.dimension.pole_a

    @property
    def negative(self) -> Category:
        return self.dimension.pole_b

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "sector": self.sector,
            "blurb": self.blurb,
            "instrument": self.instrument,
            "fictitious": self.fictitious,
            "dimension": self.dimension.to_dict(),
            "target_a": self.target_a.to_dict(),
            "target_b": self.target_b.to_dict() if self.target_b else None,
            "positive": self.positive.to_dict(),
            "negative": self.negative.to_dict(),
        }


STUDIES: dict[str, Study] = {}


def _register(study: Study) -> Study:
    STUDIES[study.id] = study
    return study


def _brand(key: str, label: str, stimuli: list[str], colour: str) -> Category:
    return Category(key=key, label=label, stimuli=stimuli, colour=colour)


# The default studies use **invented brands**. That is not squeamishness: a
# publicly reachable tool that outputs "a strong negative automatic evaluation
# of <real company>" from one participant is making a claim about a real
# business off n = 1, and no amount of small print under the chart undoes the
# screenshot. Invented brands also mirror how pack and positioning testing
# actually runs, where the stimuli are unreleased designs.
_register(Study(
    id="gin-premium",
    name="Aurelia vs Northvane — premium positioning",
    sector="Premium spirits · pack & positioning",
    blurb=(
        "Two invented gin brands, tested on the dimension a brand manager "
        "actually decides on: does the pack read as premium or as everyday? "
        "This is the comparative question a two-target IAT is the right "
        "instrument for."
    ),
    target_a=_brand("target_a", "AURELIA",
                    ["AURELIA", "Aurelia Gin", "Aurelia London Dry", "Gold-capped bottle",
                     "Aurelia No. 7"], TARGET_COLOUR),
    target_b=_brand("target_b", "NORTHVANE",
                    ["NORTHVANE", "Northvane Gin", "Northvane Dry", "Slate-labelled bottle",
                     "Northvane Reserve"], TARGET_COLOUR_B),
    dimension=DIMENSIONS["premium"],
))

_register(Study(
    id="gin-valence",
    name="Aurelia vs Northvane — plain liking",
    sector="Premium spirits · calibration run",
    blurb=(
        "The same two invented brands on the canonical positive/negative word "
        "set. Running this alongside the premium test is the instrument check: "
        "the valence version uses the exact stimuli the published literature "
        "uses, so its D is comparable to anything in print."
    ),
    target_a=_brand("target_a", "AURELIA",
                    ["AURELIA", "Aurelia Gin", "Aurelia London Dry", "Gold-capped bottle",
                     "Aurelia No. 7"], TARGET_COLOUR),
    target_b=_brand("target_b", "NORTHVANE",
                    ["NORTHVANE", "Northvane Gin", "Northvane Dry", "Slate-labelled bottle",
                     "Northvane Reserve"], TARGET_COLOUR_B),
    dimension=DIMENSIONS["valence"],
))

_register(Study(
    id="fintech-trust",
    name="Kestrel Pay vs Bastion Bank — trust",
    sector="Financial services · brand equity",
    blurb=(
        "A challenger against an incumbent on trust. Stated trust in financial "
        "brands is heavily socially desirable, which is the classic case for "
        "measuring it implicitly instead."
    ),
    target_a=_brand("target_a", "KESTREL PAY",
                    ["KESTREL PAY", "Kestrel app", "Kestrel card", "Kestrel Instant",
                     "Kestrel Pots"], TARGET_COLOUR),
    target_b=_brand("target_b", "BASTION BANK",
                    ["BASTION BANK", "Bastion branch", "Bastion Current", "Bastion Saver",
                     "Bastion Advisor"], TARGET_COLOUR_B),
    dimension=DIMENSIONS["trust"],
))

_register(Study(
    id="aurelia-sciat",
    name="Aurelia alone — single-category test",
    sector="Premium spirits · absolute read",
    blurb=(
        "One brand, no comparator. Use this when there is no fair contrast: a "
        "two-target IAT cannot tell you whether Aurelia is fast or Northvane is "
        "slow, and the competitor you pick partly decides the answer."
    ),
    target_a=_brand("target_a", "AURELIA",
                    ["AURELIA", "Aurelia Gin", "Aurelia London Dry", "Gold-capped bottle",
                     "Aurelia No. 7"], TARGET_COLOUR),
    target_b=None,
    dimension=DIMENSIONS["premium"],
))


def custom_study(
    brand_a: str,
    brand_b: str | None,
    *,
    dimension_key: str = "premium",
    sector: str = "Operator-defined",
) -> Study:
    """Build a study from brand names typed in by the operator.

    Only the brand name itself is used as a stimulus. That is weaker than a
    proper multi-exemplar set — a single exemplar means the measure partly
    reflects that one word rather than the category — and the report says so
    rather than pretending otherwise. Real brand names entered here produce a
    result about one person's response times, not about the brand.
    """
    dim = DIMENSIONS.get(dimension_key, DIMENSIONS["premium"])
    a_label = (brand_a or "").strip()[:24] or "Brand A"
    a = _brand("target_a", a_label.upper(), [a_label.upper()], TARGET_COLOUR)
    b = None
    if brand_b and brand_b.strip():
        b_label = brand_b.strip()[:24]
        b = _brand("target_b", b_label.upper(), [b_label.upper()], TARGET_COLOUR_B)
    return Study(
        id="custom",
        name=f"{a.label} vs {b.label}" if b else f"{a.label} — single-category",
        sector=sector,
        blurb=(
            "Operator-defined brands with a single exemplar per category. The "
            "measure is correspondingly narrower than a five-exemplar set and "
            "the report flags it."
        ),
        target_a=a,
        target_b=b,
        dimension=dim,
        fictitious=False,
    )


# --------------------------------------------------------------------------
# Block plans
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BlockPlan:
    index: int
    n_trials: int
    kind: Literal["target", "attribute", "combined", "sciat", "motor", "reading"]
    role: Literal["practice", "test", "sciat"] | None
    pairing: Literal["congruent", "incongruent"] | None
    reversed_targets: bool
    instruction: str


#: Trial-count presets. The full preset is the published procedure; the shorter
#: ones exist because nobody sits through 190 trials in a demo, and the tool
#: states the cost of that rather than hiding it.
PRESETS: dict[str, dict] = {
    "full": {
        "label": "Full (Greenwald 2003/2022)",
        "counts": {1: 20, 2: 20, 3: 20, 4: 40, 5: 30, 6: 20, 7: 40},
        "sciat_counts": {1: 24, 2: 72, 3: 24, 4: 72},
        "calibration": {"motor": 20, "reading": 10},
        "minutes": "6-8 min",
        "note": "The published trial counts, with the 30-trial Block 5 from the 2022 best-practice update. Use this for anything you intend to report.",
    },
    "standard": {
        "label": "Standard demo",
        "counts": {1: 12, 2: 12, 3: 12, 4: 24, 5: 18, 6: 12, 7: 24},
        "sciat_counts": {1: 12, 2: 36, 3: 12, 4: 36},
        "calibration": {"motor": 12, "reading": 6},
        "minutes": "3-4 min",
        "note": "About 60% of the published trial count. D stays unbiased but its confidence interval widens by roughly a third.",
    },
    "express": {
        "label": "Express",
        "counts": {1: 8, 2: 8, 3: 8, 4: 16, 5: 12, 6: 8, 7: 16},
        "sciat_counts": {1: 8, 2: 24, 3: 8, 4: 24},
        "calibration": {"motor": 8, "reading": 4},
        "minutes": "2 min",
        "note": "Demonstration only. Too few trials for a stable individual estimate — read the D as illustrative, not as a measurement.",
    },
}

#: Phrases for the reading-speed calibration, spanning a range of word counts so
#: a per-participant reading slope can be estimated rather than a single mean.
READING_PHRASES = [
    "Bottle",
    "Green bottle",
    "A tall green bottle",
    "A tall green bottle on a shelf",
    "A tall green bottle standing on a wooden shelf",
    "A tall green bottle standing on a wooden shelf beside the window",
    "Label",
    "Gold label",
    "A small gold label",
    "A small gold label printed with fine lettering",
    "A small gold label printed with fine lettering around the neck",
    "A small gold label printed with fine lettering around the neck of the bottle",
]


def calibration_blocks(preset: str) -> list[BlockPlan]:
    cal = PRESETS[preset]["calibration"]
    return [
        BlockPlan(-2, cal["motor"], "motor", None, None, False,
                  "Warm-up. An arrow will point left or right. Press E for left and I for "
                  "right, as fast as you can. This measures how quickly you can move on "
                  "this device — it is not part of the result."),
        BlockPlan(-1, cal["reading"], "reading", None, None, False,
                  "Reading check. A phrase will appear. Press I the instant you have read "
                  "it. This measures your reading speed so a longer word is not mistaken "
                  "for a weaker association."),
    ]


def iat_blocks(preset: str, congruent_first: bool) -> list[BlockPlan]:
    """The seven-block IAT.

    ``congruent_first`` is the counterbalancing factor. Half of participants
    should meet the congruent pairing first and half the incongruent pairing,
    because whichever pairing is encountered second carries a practice-driven
    advantage. Failing to counterbalance it does not add noise — it adds bias,
    in a direction fixed by the researcher.
    """
    counts = PRESETS[preset]["counts"]
    first: Literal["congruent", "incongruent"] = "congruent" if congruent_first else "incongruent"
    second: Literal["congruent", "incongruent"] = "incongruent" if congruent_first else "congruent"

    return [
        BlockPlan(1, counts[1], "target", None, None, False,
                  "Sort the brands. Press E for the brand on the left and I for the one on the right."),
        BlockPlan(2, counts[2], "attribute", None, None, False,
                  "Now sort the words by which group they belong to."),
        BlockPlan(3, counts[3], "combined", "practice", first, False,
                  "Both tasks together — the categories now share keys. Go as fast as you can without guessing."),
        BlockPlan(4, counts[4], "combined", "test", first, False,
                  "Same pairing, more trials. Keep your fingers resting on E and I."),
        BlockPlan(5, counts[5], "target", None, None, True,
                  "The brands have swapped sides. Sort them again with the new arrangement."),
        BlockPlan(6, counts[6], "combined", "practice", second, True,
                  "Both tasks together with the new pairing."),
        BlockPlan(7, counts[7], "combined", "test", second, True,
                  "Final block. Same pairing, more trials."),
    ]


def sciat_blocks(preset: str, congruent_first: bool) -> list[BlockPlan]:
    """Four SC-IAT blocks: a practice and a critical block for each pairing.

    Only the critical blocks carry a ``role``, so only they are scored — which
    is what Karpinski & Steinman's procedure requires.
    """
    counts = PRESETS[preset]["sciat_counts"]
    first: Literal["congruent", "incongruent"] = "congruent" if congruent_first else "incongruent"
    second: Literal["congruent", "incongruent"] = "incongruent" if congruent_first else "congruent"
    return [
        BlockPlan(1, counts[1], "sciat", None, first, False,
                  "Practice. Sort each item to the left or right key. Not scored."),
        BlockPlan(2, counts[2], "sciat", "sciat", first, False,
                  "Scored block. Respond within the time window — you will be prompted if you are too slow."),
        BlockPlan(3, counts[3], "sciat", None, second, True,
                  "The brand has moved to the other key. Practise the new arrangement. Not scored."),
        BlockPlan(4, counts[4], "sciat", "sciat", second, True,
                  "Final scored block."),
    ]


# --------------------------------------------------------------------------
# Trial sequence generation
# --------------------------------------------------------------------------


@dataclass(slots=True)
class GeneratedTrial:
    index: int
    block_index: int
    block_role: str | None
    pairing: str | None
    stimulus: str
    stimulus_category: str
    correct_key: str
    correct_side: Side
    word_count: int = 1

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "block_index": self.block_index,
            "block_role": self.block_role,
            "pairing": self.pairing,
            "stimulus": self.stimulus,
            "stimulus_category": self.stimulus_category,
            "correct_key": self.correct_key,
            "correct_side": self.correct_side,
            "word_count": self.word_count,
        }


def _cycle_sample(pool: list[str], n: int, rng: np.random.Generator) -> list[str]:
    """Sample ``n`` items, exhausting the pool before repeating any of them.

    Sampling with replacement would let one exemplar appear five times in a row
    by chance, which turns a category measure into a measure of that exemplar.
    """
    out: list[str] = []
    while len(out) < n:
        block = list(pool)
        rng.shuffle(block)
        out.extend(block)
    return out[:n]


def _no_long_runs(seq: list[str], rng: np.random.Generator, max_run: int = 3) -> list[str]:
    """Break runs of the same category longer than ``max_run``.

    Long runs let a participant stop reading and repeat the last keypress, which
    produces fast, correct and meaningless trials.
    """
    seq = list(seq)
    for _ in range(200):
        run, worst = 1, -1
        for i in range(1, len(seq)):
            run = run + 1 if seq[i] == seq[i - 1] else 1
            if run > max_run:
                worst = i
                break
        if worst == -1:
            return seq
        j = int(rng.integers(0, len(seq)))
        seq[worst], seq[j] = seq[j], seq[worst]
    return seq


def build_session(
    study: Study,
    *,
    preset: str = "standard",
    seed: int | None = None,
    include_calibration: bool = True,
) -> dict:
    """Generate a complete, enumerated session for the browser to run."""
    if preset not in PRESETS:
        preset = "standard"

    seed_used = int(seed) if seed is not None else int(
        np.random.default_rng().integers(0, 2**31 - 1)
    )
    rng = np.random.default_rng(seed_used)

    # Counterbalancing factor 1: which pairing is met first.
    congruent_first = bool(rng.integers(0, 2))
    # Counterbalancing factor 2: which side target A starts on. Cosmetic for D,
    # but it removes any handedness confound from the raw latencies.
    a_starts_left = bool(rng.integers(0, 2))

    is_iat = study.instrument == "iat"
    blocks: list[BlockPlan] = []
    if include_calibration:
        blocks.extend(calibration_blocks(preset))
    blocks.extend(
        iat_blocks(preset, congruent_first) if is_iat else sciat_blocks(preset, congruent_first)
    )

    pole_a, pole_b = study.dimension.pole_a, study.dimension.pole_b
    trials: list[GeneratedTrial] = []
    idx = 0
    block_meta: list[dict] = []

    for plan in blocks:
        left_cats: list[Category] = []
        right_cats: list[Category] = []

        # ---- Calibration blocks -----------------------------------------
        if plan.kind == "motor":
            left_cats = [Category("motor_left", "◀  Left", ["◀"], "#f4f4f5")]
            right_cats = [Category("motor_right", "Right  ▶", ["▶"], "#f4f4f5")]
            seq = _no_long_runs(
                _cycle_sample(["motor_left", "motor_right"], plan.n_trials, rng), rng
            )
            for cat_key in seq:
                side: Side = "left" if cat_key == "motor_left" else "right"
                trials.append(GeneratedTrial(
                    index=idx, block_index=plan.index, block_role=None, pairing=None,
                    stimulus="◀" if side == "left" else "▶",
                    stimulus_category=cat_key,
                    correct_key=LEFT_KEY if side == "left" else RIGHT_KEY,
                    correct_side=side,
                ))
                idx += 1

        elif plan.kind == "reading":
            left_cats = [Category("read", "—", [], "#f4f4f5")]
            right_cats = [Category("read", "Press I when read", [], "#f4f4f5")]
            phrases = _cycle_sample(READING_PHRASES, plan.n_trials, rng)
            for phrase in phrases:
                trials.append(GeneratedTrial(
                    index=idx, block_index=plan.index, block_role=None, pairing=None,
                    stimulus=phrase, stimulus_category="reading",
                    correct_key=RIGHT_KEY, correct_side="right",
                    word_count=len(phrase.split()),
                ))
                idx += 1

        else:
            # ---- Task blocks ---------------------------------------------
            a_left = a_starts_left if not plan.reversed_targets else not a_starts_left
            positive_with_a = plan.pairing != "incongruent"

            if is_iat:
                if plan.kind == "target":
                    left_cats = [study.target_a if a_left else study.target_b]
                    right_cats = [study.target_b if a_left else study.target_a]
                elif plan.kind == "attribute":
                    left_cats, right_cats = [pole_a], [pole_b]
                else:
                    positive_on_left = (a_left and positive_with_a) or (not a_left and not positive_with_a)
                    left_cats = [study.target_a if a_left else study.target_b]
                    right_cats = [study.target_b if a_left else study.target_a]
                    left_cats.append(pole_a if positive_on_left else pole_b)
                    right_cats.append(pole_b if positive_on_left else pole_a)
            else:
                # SC-IAT: one target, two attribute poles. The target sits with
                # pole A in the congruent block and with pole B otherwise.
                if positive_with_a:
                    left_cats, right_cats = [study.target_a, pole_a], [pole_b]
                else:
                    left_cats, right_cats = [pole_a], [study.target_a, pole_b]

            left_keys = {c.key for c in left_cats}

            if is_iat and plan.kind == "combined":
                # Combined blocks alternate a target trial and an attribute
                # trial. Without alternation a participant can batch-process one
                # dimension, and the association being measured degrades into
                # two separate sorting tasks.
                n = plan.n_trials
                t_seq = _no_long_runs(_cycle_sample(["target_a", "target_b"], (n + 1) // 2, rng), rng)
                a_seq = _no_long_runs(_cycle_sample([pole_a.key, pole_b.key], n // 2, rng), rng)
                cat_seq = [t_seq[i // 2] if i % 2 == 0 else a_seq[i // 2] for i in range(n)]
            elif is_iat and plan.kind == "target":
                cat_seq = _no_long_runs(_cycle_sample(["target_a", "target_b"], plan.n_trials, rng), rng)
            elif is_iat and plan.kind == "attribute":
                cat_seq = _no_long_runs(_cycle_sample([pole_a.key, pole_b.key], plan.n_trials, rng), rng)
            else:
                # SC-IAT 7:7:10 ratio. Whichever pole shares a key with the
                # target gets 7 parts, the lone pole gets 10, so both keys are
                # correct about equally often. Without this correction one key
                # is right more than half the time and participants exploit it.
                if positive_with_a:
                    ratio = ["target_a"] * 7 + [pole_a.key] * 7 + [pole_b.key] * 10
                else:
                    ratio = ["target_a"] * 7 + [pole_b.key] * 7 + [pole_a.key] * 10
                cat_seq = _no_long_runs(_cycle_sample(ratio, plan.n_trials, rng), rng)

            by_key = {
                c.key: c for c in
                (study.target_a, study.target_b, pole_a, pole_b) if c is not None
            }
            # sorted(), not set() — iteration order over a set of strings varies
            # with the process hash seed, so an unsorted loop here would consume
            # the RNG in a different order on every run and the same seed would
            # produce a different session. That would quietly void the whole
            # reproducibility claim while every within-process test still passed.
            pools = {
                k: _cycle_sample(by_key[k].stimuli, cat_seq.count(k), rng)
                for k in sorted(set(cat_seq))
            }
            cursors = {k: 0 for k in pools}

            for cat_key in cat_seq:
                stim = pools[cat_key][cursors[cat_key]]
                cursors[cat_key] += 1
                side = "left" if cat_key in left_keys else "right"
                trials.append(GeneratedTrial(
                    index=idx, block_index=plan.index, block_role=plan.role,
                    pairing=plan.pairing if plan.role else None,
                    stimulus=stim, stimulus_category=cat_key,
                    correct_key=LEFT_KEY if side == "left" else RIGHT_KEY,
                    correct_side=side, word_count=len(stim.split()),
                ))
                idx += 1

        block_meta.append({
            "index": plan.index,
            "n_trials": plan.n_trials,
            "kind": plan.kind,
            "role": plan.role,
            "pairing": plan.pairing,
            "instruction": plan.instruction,
            "left": [c.to_dict() for c in left_cats],
            "right": [c.to_dict() for c in right_cats],
            "response_window_ms": 1500 if (not is_iat and plan.kind == "sciat") else None,
            "scored": plan.role is not None,
        })

    task_blocks = [b for b in block_meta if b["kind"] not in ("motor", "reading")]

    return {
        "study": study.to_dict(),
        "instrument": study.instrument,
        "preset": preset,
        "preset_label": PRESETS[preset]["label"],
        "preset_note": PRESETS[preset]["note"],
        "estimated_minutes": PRESETS[preset]["minutes"],
        "seed": seed_used,
        "counterbalance": {
            "congruent_pairing_first": congruent_first,
            "target_a_starts_left": a_starts_left,
            "note": (
                "Both factors are drawn from the session seed. Re-running with the "
                "same seed reproduces this exact sequence, stimulus for stimulus."
            ),
        },
        "blocks": block_meta,
        "n_task_blocks": len(task_blocks),
        "trials": [t.to_dict() for t in trials],
        "n_trials": len(trials),
        "n_scored_trials": sum(1 for t in trials if t.block_role),
        "keys": {"left": LEFT_KEY, "right": RIGHT_KEY},
    }
