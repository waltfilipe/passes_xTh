"""Metric groups for the Comparação tab (shared by app and engine)."""

from __future__ import annotations

COMPARISON_IMPACT_KEYS: tuple[str, ...] = (
    "impact_passes_p90",
    "phi_p90",
    "long_impact_passes",
    "impact_passes",
    "high_impact_passes",
    "aggression_aip",
)

COMPARISON_PROGRESSION_KEYS: tuple[str, ...] = (
    "progressive_passes_p90",
    "final_third_passes_p90",
    "long_balls",
    "progressive_passes",
    "final_third_passes",
    "key_passes",
)

COMPARISON_CARD_GROUPS: dict[str, tuple[str, ...]] = {
    "comparison_impact": COMPARISON_IMPACT_KEYS,
    "comparison_progression": COMPARISON_PROGRESSION_KEYS,
}
