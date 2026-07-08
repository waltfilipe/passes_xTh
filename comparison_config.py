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

IMPACT_MODEL_DEFAULT = "atual"
IMPACT_MODEL_OPT1_SHORT_FT = "opt1_short_ft"

IMPACT_MODEL_LABELS: dict[str, str] = {
    IMPACT_MODEL_DEFAULT: "Atual (ganho relativo)",
    IMPACT_MODEL_OPT1_SHORT_FT: "Opção 1 + via curta",
}


def normalize_impact_model(model: str | None) -> str:
    key = str(model or IMPACT_MODEL_DEFAULT).strip().lower()
    return key if key in IMPACT_MODEL_LABELS else IMPACT_MODEL_DEFAULT
