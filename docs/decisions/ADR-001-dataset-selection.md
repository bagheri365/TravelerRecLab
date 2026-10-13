# ADR-001 — Primary dataset selection

**Status:** Accepted
**Date:** 2026-10-13

## Problem
Learn and test traveler representations from publicly accessible hotel-search behavior.

## Options
1. **Trivago RecSys Challenge 2019**: ordered sessions, actions, candidate lists on click-outs, prices and item metadata; no confirmed booking outcome.
2. Expedia Hotel Recommendations: click/booking indicators and hotel clusters, but no complete impression and action sequence for a unified funnel.
3. Expedia Personalized Sort: candidate-level click/booking ranking, less suitable for session event modeling.

## Decision and why
Use Trivago `train.csv` for sequence representation experiments and keep Expedia datasets for **separate** supervised studies. Do not join travelers or items across datasets.

## Trade-offs / threats
Trivago click-outs are not bookings, dataset is historical, and data collection/position effects can bias outcomes. Check distribution before choosing architectures. Do not redistribute raw data without permission.

## Evidence
Local `inspect_trivago` output from the project owner, 2026-10-13 (statistics in [dataset audit](../data/trivago_audit_2026-10-13.md)). The audit is descriptive, not proof of downstream relevance.

## Revisit if
A suitable public dataset is found with time-ordered search, impression, click, and booking events in the same identity space.

## Interview explanation (60 seconds)
> I chose Trivago because it captures typed, ordered travel interactions—the key input for traveler sequence modeling. I explicitly separate click-out recommendation from booking prediction because the dataset doesn't show confirmed bookings. I'd use different data to validate a booking model rather than mislabel click-outs as conversions.
