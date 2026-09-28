# 01 - Problem statement (SIH26078)

## The problem

Extreme weather events cause the majority of weather-related losses in India,
and the response to them is limited less by computation than by **timeliness
and by trust**:

- Warnings arrive as isolated text products with no machine-readable history,
  so they cannot be analysed, compared or integrated with other data.
- Extreme events are *tracked* phenomena, not point measurements. A warning for
  one location says nothing about where the system is three hours later.
- Deterministic numerical models are expensive to run and hard to validate at
  the local scale where decisions are actually made.
- Every public dashboard that claims forecast skill has, in general, not shown
  its work: metrics appear without their provenance, and an unavailable number
  is quietly rendered as zero.

## What this project sets out to do

Build an end-to-end pipeline that ingests NWP-style and historical weather
data, validates it consistently, detects extreme anomalies against a
climatology, tracks those events through time, applies graph and convolutional
models to them, and serves the result through an API and an interactive
dashboard — with every measurement carrying its provenance and every
unavailable quantity reported as unavailable.

## What this project explicitly does not claim

This is the part that distinguishes it from most submissions of this kind.

- **No forecast skill is demonstrated.** No verified ground-truth warning
  catalogue was available, so no accuracy, precision, recall or F1 figure is
  reported. Where such a number would normally appear, this system prints
  `unavailable` and the reason.
- **No real-time capability.** The system ingests historical reanalysis or
  locally generated synthetic data. "Near-real-time" describes the
  orchestration loop, not the data.
- **No disaster-prevention claim.** Nothing here guarantees evacuation, saves
  lives, or reduces economic loss. No such figure has been measured.
- **The alert severities are this project's own categories**, not official IMD
  or WMO warnings, and must not be used for emergency decisions.
- **The models are research prototypes.** On the available data the GNN
  overfits visibly, and the AI downscaler does not beat cubic interpolation.
  Both results are reported rather than concealed.

The deliverable is therefore an **honest, reproducible engineering
foundation** with clearly marked prototype components — not a claim of
operational capability. See `07_limitations.md` for the full list.

## Why the honesty is the point

An early-warning system is infrastructure. Infrastructure that overstates its
reliability is worse than no system at all, because decisions get made on the
overstatement. Every design decision in this project follows from that: a
missing model is `SKIPPED` with a reason rather than replaced with a default, a
metric that could not be computed is `null` rather than `0`, and the
reproducibility check excludes only the fields that are *genuinely* expected to
vary.
