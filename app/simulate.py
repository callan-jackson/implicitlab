"""Synthetic respondents.

Two reasons this exists, and neither is padding.

**Testing.** Asserting that the pipeline runs is easy; asserting that it
*recovers the truth* needs data with a known answer. The generator takes a true
effect size and produces trials from it, so a test can check that the D coming
out of the scorer is the D that went in.

**Demonstration.** A full IAT is 190 trials. Nobody watching a demo wants to sit
through that before seeing the dashboard, and a dashboard with no data in it
demonstrates nothing. Simulated sessions are labelled as simulated everywhere
they appear — in the API response, on the dashboard, and in the export — because
the one thing worse than no data in a research tool is data whose provenance is
unclear.

Latencies are drawn from an **ex-Gaussian** distribution: a normal component
convolved with an exponential tail. That is the standard descriptive model for
reaction times, and it matters here because it is right-skewed. Simulating from
a plain normal would produce data on which a t-test looks fine and the
resampling machinery looks like overkill — the simulator would be quietly
flattering the analysis.
"""

from __future__ import annotations

import numpy as np

from .design import build_session

# Parameters roughly matching published IAT latencies on a desktop keyboard:
# a mode around 550 ms, a median near 650 ms, and a long right tail.
MU_BASE = 470.0        # normal component mean, ms
SIGMA_BASE = 70.0      # normal component SD, ms
TAU_BASE = 190.0       # exponential component mean, ms


def _exgauss(rng: np.random.Generator, n: int, mu: float, sigma: float, tau: float) -> np.ndarray:
    return rng.normal(mu, sigma, n) + rng.exponential(tau, n)


def simulate_session(
    *,
    study,
    preset: str = "standard",
    seed: int = 1,
    true_d: float = 0.4,
    base_rt_ms: float = MU_BASE,
    error_rate: float = 0.06,
    speed_variability: float = 1.0,
    careless: bool = False,
) -> tuple[dict, list[dict], dict]:
    """Generate a design and a plausible set of responses to it.

    Parameters
    ----------
    true_d:
        The effect to build in. Positive means faster on the congruent pairing.
        The realised D will not equal this exactly — that is the point; sampling
        noise at these trial counts is substantial and the simulation shows it.
    careless:
        If set, produce a respondent who is button-mashing: very fast, high
        error rate. Used to check that the exclusion rules actually fire.
    """
    design = build_session(study, preset=preset, seed=seed)
    rng = np.random.default_rng(seed + 977)

    # Convert the target D into a latency gap. D divides by the inclusive SD,
    # which for these parameters lands around 200 ms, so the gap is roughly
    # true_d * 200.
    inclusive_sd_estimate = np.sqrt(SIGMA_BASE**2 + TAU_BASE**2) * speed_variability
    gap_ms = true_d * inclusive_sd_estimate

    sigma = SIGMA_BASE * speed_variability
    tau = TAU_BASE * speed_variability

    records: list[dict] = []
    for t in design["trials"]:
        cat = t["stimulus_category"]

        if cat.startswith("motor"):
            # A simple cued keypress: much faster than a categorisation, and
            # with a shorter tail.
            lat = float(_exgauss(rng, 1, 230.0, 35.0, 60.0)[0])
            correct = rng.random() > 0.02
        elif cat == "reading":
            # Reading time scales with word count. A ~55 ms/word slope plus a
            # fixed decision cost is in the right region for silent reading.
            lat = float(_exgauss(rng, 1, 300.0 + 55.0 * t.get("word_count", 1), 60.0, 90.0)[0])
            correct = True
        else:
            mu = base_rt_ms
            if t["pairing"] == "incongruent":
                mu += gap_ms / 2
            elif t["pairing"] == "congruent":
                mu -= gap_ms / 2
            # Single-discrimination blocks are easier than combined ones.
            if t["block_role"] is None:
                mu -= 60.0
            lat = float(_exgauss(rng, 1, mu, sigma, tau)[0])
            # Errors are more likely in the harder pairing, which is the real
            # speed-accuracy trade-off the error-rate check looks for.
            p_err = error_rate * (1.6 if t["pairing"] == "incongruent" else 1.0)
            correct = rng.random() > p_err

        if careless:
            lat = float(rng.uniform(150, 320))
            correct = rng.random() > 0.45

        lat = max(120.0, lat)
        response_key = t["correct_key"] if correct else ("I" if t["correct_key"] == "E" else "E")
        # Under forced correction an error costs an extra keypress.
        to_correct = lat if correct else lat + float(_exgauss(rng, 1, 420.0, 90.0, 120.0)[0])

        records.append({
            "index": t["index"],
            "block_index": t["block_index"],
            "block_role": t["block_role"],
            "pairing": t["pairing"],
            "stimulus": t["stimulus"],
            "stimulus_category": cat,
            "correct_key": t["correct_key"],
            "response_key": response_key,
            "latency_ms": round(lat, 3),
            "latency_to_correct_ms": round(to_correct, 3),
            "correct": correct,
            "timed_out": False,
            "n_corrections": 0 if correct else 1,
            "onset_uncertainty_ms": 8.33,
            "dispatch_delay_ms": round(float(rng.uniform(0.2, 3.5)), 3),
            "used_event_timestamp": True,
            "focus_lost": False,
            "word_count": t.get("word_count", 1),
        })

    client_meta = {
        "simulated": True,
        "simulated_true_d": true_d,
        "refresh_hz": 60.0,
        "onset_uncertainty_ms": 8.33,
        "median_dispatch_delay_ms": 1.6,
        "used_event_timestamp": True,
        "focus_losses": 0,
        "note": (
            "Synthetic respondent. Latencies were drawn from an ex-Gaussian "
            "distribution with a known effect built in. This is simulated data "
            "and is labelled as such wherever it appears."
        ),
    }
    return design, records, client_meta
