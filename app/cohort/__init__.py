"""Cohort analysis: many participants, many attributes, one analytic sample.

The single-session pipeline in :mod:`app.analysis` answers "what did this
person do". A commercial study never asks that question. It recruits a panel,
runs every respondent through a battery of brand-attribute tests, and asks
which associations hold *across the audience* and *between segments of it*.

The package is laid out in the order the data flows:

    ingest     raw CSV / JSON dump -> validated long-format trial table
    scoring    participant screening + vectorised D per (participant, attribute)
    stats      group-level inference: bootstrap CIs, parametric and permutation
               tests, false-discovery-rate correction
    analysis   parameters in, the complete cohort report out
    takeaways  deterministic executive bullet points
    charts     matplotlib figures for the deliverables
    export_*   Excel workbook and PowerPoint deck

Everything downstream of ``ingest`` is a pure function of (batch, parameters),
so the numbers on screen, in the workbook and in the deck are the same numbers.
"""
