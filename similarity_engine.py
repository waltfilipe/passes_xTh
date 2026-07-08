"""Série B ↔ Série A player similarity (options A and C)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from heuristic_scoring import POSITION_GROUPS_ORDER, position_group

# Option A — percentile profile within league + position group.
SIMILARITY_METRICS_A: tuple[str, ...] = (
    "impact_passes_p90",
    "phi_p90",
    "impact_per_pass",
    "dxt_per_pass",
    "progressive_passes_p90",
    "final_third_passes_p90",
    "key_passes",
    "long_impact_per_long_pass",
    "construction_aip_per_pass",
    "aggression_aip_per_pass",
)

# Option C — z-score distance (higher weight on core impact volume).
SIMILARITY_METRICS_C: tuple[str, ...] = SIMILARITY_METRICS_A
SIMILARITY_WEIGHTS_C: dict[str, float] = {
    "impact_passes_p90": 2.0,
    "phi_p90": 2.0,
    "dxt_p90": 2.0,
    "impact_per_pass": 1.5,
    "dxt_per_pass": 1.5,
    "phi_per_pass": 1.5,
    "progressive_passes_p90": 1.0,
    "final_third_passes_p90": 1.0,
    "key_passes": 1.0,
    "long_impact_per_long_pass": 1.0,
    "construction_aip_per_pass": 1.0,
    "aggression_aip_per_pass": 1.0,
}

SERIE_A_POSITION_TO_GROUP: dict[str, str] = {
    "CB": "Zagueiros",
    "CM": "Meio-campistas",
    "ST": "Atacantes",
}

# Série A CSV só tem CB/CM/ST — laterais e extremos da Série B usam pool de meias.
SERIE_A_SEARCH_GROUP: dict[str, str] = {
    "Zagueiros": "Zagueiros",
    "Laterais": "Meio-campistas",
    "Meio-campistas": "Meio-campistas",
    "Extremos": "Meio-campistas",
    "Atacantes": "Atacantes",
}

TOP_K_DEFAULT = 5
MIN_PASSES_SERIE_A = 100


def _metric_vector(player: dict, keys: tuple[str, ...]) -> np.ndarray:
    return np.array([float(player.get(k) or 0.0) for k in keys], dtype=float)


def _fill_missing(values: np.ndarray) -> np.ndarray:
    out = values.copy()
    mask = ~np.isfinite(out)
    if mask.any():
        out[mask] = 0.0
    return out


def _percentile_table(players: list[dict], keys: tuple[str, ...]) -> pd.DataFrame:
    rows = []
    for p in players:
        row = {"player_id": p["player_id"]}
        for k in keys:
            row[k] = float(p.get(k) or 0.0)
        rows.append(row)
    df = pd.DataFrame(rows).set_index("player_id")
    pct = df.rank(pct=True, method="average") * 100.0
    return pct


def _zscore_table(players: list[dict], keys: tuple[str, ...]) -> pd.DataFrame:
    rows = []
    for p in players:
        row = {"player_id": p["player_id"]}
        for k in keys:
            row[k] = float(p.get(k) or 0.0)
        rows.append(row)
    df = pd.DataFrame(rows).set_index("player_id")
    mean = df.mean()
    std = df.std(ddof=0).replace(0, np.nan)
    z = (df - mean) / std
    return z.fillna(0.0)


def _distance_to_similarity(dist: float, scale: float) -> float:
    if scale <= 0:
        return 100.0 if dist == 0 else 0.0
    return float(np.clip(100.0 * (1.0 - dist / scale), 0.0, 100.0))


def find_similar_option_a(
    target: dict,
    pool: list[dict],
    *,
    top_k: int = TOP_K_DEFAULT,
) -> list[dict[str, Any]]:
    """Percentile neighbours within the same Série A search group."""
    if not pool:
        return []
    keys = SIMILARITY_METRICS_A
    pct_pool = _percentile_table(pool, keys)
    target_pct = {}
    for k in keys:
        val = float(target.get(k) or 0.0)
        col = pct_pool[k]
        target_pct[k] = float((col < val).mean() * 100.0) if len(col) else 50.0
    tvec = np.array([target_pct[k] for k in keys], dtype=float)

    scale = float(np.sqrt(len(keys)) * 100.0)
    results = []
    for cand in pool:
        if cand["player_id"] == target.get("player_id"):
            continue
        cvec = pct_pool.loc[cand["player_id"]].to_numpy(dtype=float)
        dist = float(np.linalg.norm(_fill_missing(tvec - cvec)))
        results.append({
            **cand,
            "similarity_pct": round(_distance_to_similarity(dist, scale), 1),
            "distance": round(dist, 3),
        })
    results.sort(key=lambda r: (-r["similarity_pct"], r["distance"]))
    return results[:top_k]


def find_similar_option_c(
    target: dict,
    pool: list[dict],
    *,
    top_k: int = TOP_K_DEFAULT,
) -> list[dict[str, Any]]:
    """Z-score neighbours (per league pool) with weighted Euclidean distance."""
    if not pool:
        return []
    keys = SIMILARITY_METRICS_C
    weights = np.array([SIMILARITY_WEIGHTS_C.get(k, 1.0) for k in keys], dtype=float)
    z_pool = _zscore_table(pool, keys)

    tvec = _metric_vector(target, keys)
    mean = z_pool.mean()
    std = z_pool.std(ddof=0).replace(0, np.nan)
    tz = ((pd.Series(tvec, index=keys) - mean) / std).fillna(0.0).to_numpy(dtype=float)

    diffs = z_pool.to_numpy(dtype=float) - tz
    dists = np.sqrt((diffs ** 2 * weights).sum(axis=1))
    scale = float(dists.max()) if len(dists) else 1.0
    if scale <= 0:
        scale = 1.0

    results = []
    for dist, cand in zip(dists, pool):
        if cand["player_id"] == target.get("player_id"):
            continue
        results.append({
            **cand,
            "similarity_pct": round(_distance_to_similarity(float(dist), scale), 1),
            "distance": round(float(dist), 3),
        })
    results.sort(key=lambda r: (-r["similarity_pct"], r["distance"]))
    return results[:top_k]


def serie_a_search_group(sb_group: str | None) -> str | None:
    if not sb_group:
        return None
    return SERIE_A_SEARCH_GROUP.get(sb_group)


def group_serie_a_pool(players: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {g: [] for g in POSITION_GROUPS_ORDER}
    for p in players:
        grp = p.get("position_group")
        if grp in out:
            out[str(grp)].append(p)
    return out
