"""Metric groups and impact-model config (shared by app and engine)."""

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

# Slicer 1 — ajustes de classificação (distância + via curta no terço final).
CLASSIFICATION_MODEL_DEFAULT = "atual"
CLASSIFICATION_MODEL_OPT1_SHORT_FT = "opt1_short_ft"

CLASSIFICATION_MODEL_LABELS: dict[str, str] = {
    CLASSIFICATION_MODEL_DEFAULT: "Atual (ganho relativo)",
    CLASSIFICATION_MODEL_OPT1_SHORT_FT: "Opção 1 + via curta",
}

# Slicer 2 — limiares de impact / high impact.
TIER_MODEL_DEFAULT = "atual"
TIER_MODEL_FIXED_30_50 = "fixed_30_50"
TIER_MODEL_PERCENTILE_P70_P90 = "percentile_p70_p90"

TIER_MODEL_LABELS: dict[str, str] = {
    TIER_MODEL_DEFAULT: "Atual (0,30 / 0,62)",
    TIER_MODEL_FIXED_30_50: "Fixo (0,30 / 0,50)",
    TIER_MODEL_PERCENTILE_P70_P90: "Percentil (p70 / p90)",
}

# Backward-compatible aliases (tier model only).
IMPACT_MODEL_DEFAULT = TIER_MODEL_DEFAULT
IMPACT_MODEL_FIXED_30_50 = TIER_MODEL_FIXED_30_50
IMPACT_MODEL_PERCENTILE_P70_P90 = TIER_MODEL_PERCENTILE_P70_P90
IMPACT_MODEL_LABELS = TIER_MODEL_LABELS


def normalize_classification_model(model: str | None) -> str:
    key = str(model or CLASSIFICATION_MODEL_DEFAULT).strip().lower()
    return key if key in CLASSIFICATION_MODEL_LABELS else CLASSIFICATION_MODEL_DEFAULT


def normalize_tier_model(model: str | None) -> str:
    key = str(model or TIER_MODEL_DEFAULT).strip().lower()
    return key if key in TIER_MODEL_LABELS else TIER_MODEL_DEFAULT


def normalize_impact_model(model: str | None) -> str:
    """Alias for normalize_tier_model (legacy imports)."""
    return normalize_tier_model(model)
