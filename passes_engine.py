"""Pass analytics engine: xT v4, metrics, rating (no Streamlit)."""

from __future__ import annotations

import functools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPTS = Path(__file__).resolve().parent / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from external_models import load_markov_model
from heuristic_scoring import POSITION_GROUPS_ORDER, is_outfield_position, position_group

try:
    from sofascore_positions import normalize_sofascore_position
except ImportError:
    def normalize_sofascore_position(raw, *, default: str = "CM") -> str:
        text = str(raw).strip().upper() if raw is not None else ""
        return text or default

# ── Paths & eligibility ─────────────────────────────────────────────────────
SEASON_ALL_CSV_PATH = Path(__file__).resolve().parent / "season_all_serieb.csv"
PLAYER_MATCH_STATS_PATH = Path(__file__).resolve().parent / "player_match_stats.csv"
DATA_CACHE_VERSION = 14

MIN_MINUTES_PCT = 0.30
RATING_MIN_MINUTES_PCT = 0.30
RANKING_TOP_N = 20
RATING_TOP_N = 20
RATING_SCORE_BEST = 1.0
RATING_SCORE_WORST = 0.5

# ── Pitch & zones ───────────────────────────────────────────────────────────
FIELD_X, FIELD_Y = 120.0, 80.0
HALF_LINE_X = FIELD_X / 2
FINAL_THIRD_LINE_X = 80.0
GOAL_X, GOAL_Y = 120.0, 40.0
WYSCOUT_PITCH_SIZE = 100.0
PASS_AGGRESSION_X_MIN = FIELD_X - 30.0
LONG_PASS_MIN_DISTANCE_M = 30.0
DXT_IMPACT_THRESHOLD = 0.1
DEFAULT_PLAYER_POSITION = "CM"

WYSCOUT_PROG_OWN_HALF = 30.0
WYSCOUT_PROG_CROSS_HALF = 15.0
WYSCOUT_PROG_OPP_HALF = 10.0
IMPACT_PASS_MIN_GOAL_APPROACH_FINAL_THIRD = 5.0
IMPACT_PASS_MIN_GOAL_APPROACH_REST = 10.0

# ── xT v4 classification thresholds ─────────────────────────────────────────
XT_V3_PROG_FLOOR_CLASS = 0.12
XT_V3_PROG_SCALE_CLASS = 0.19
# Pass impact (tier 1) is 5% stricter; high impact (tier 2) unchanged.
IMPACT_PROG_STRICTNESS = 1.05
XT_V3_HIGH_FLOOR_CLASS = 0.26
XT_V3_HIGH_SCALE_CLASS = 0.45

# ── xT v4 surface ───────────────────────────────────────────────────────────
XT_V3_FINE_NX, XT_V3_FINE_NY = 96, 64
XT_V3_DEF_MAX, XT_V3_MID_MAX, XT_V3_ATT_BYLINE = 0.25, 0.60, 0.94
XT_V3_SURFACE_MAX = 1.02
OPT_ATTACKING_TWO_THIRDS_X = 40.0
XT_V31_ZONE_BLEND_WIDTH = 48.0
XT_V31_LAT_DISC_MAX = 0.06
XT_V31_LAT_GATE_X = HALF_LINE_X
XT_V31_GAUSS_SIGMA_X, XT_V31_GAUSS_SIGMA_Y = 3.5, 0.0
XT_V31_COL_SMOOTH_KERNEL = (0.22, 0.56, 0.22)
XT_V31_MAX_COL_STEP_DEF, XT_V31_MAX_COL_STEP_ATT = 0.050, 0.078
XT_V31_ATT_COL_START = 10
XT_V3_LAT_CURVE_POWER = 1.0
XT_V4_MARKOV_BONUS_MAX, XT_V4_MARKOV_BONUS_POWER = 0.052, 1.0
XT_V4_MARKOV_DEF_MID_FLOOR, XT_V4_MARKOV_GATE_BLEND = 0.06, 14.0
XT_V4_SURFACE_MAX = 1.02
XT_V4_SHORT_PASS_DIST, XT_V4_SHORT_PASS_FACTOR = 8.0, 0.55
XT_V3_NEG_PENALTY_FACTOR = 0.55
XT_V3_PRESSURE_ESCAPE_BONUS = 0.02
XT_V3_PRESSURE_X_MAX = 50.0
XT_V3_WIDE_FRAC = 0.60
XT_V3_NEG_RECYCLE_X_MAX = 60.0
XT_V5_MAX_DELTA_DEF, XT_V5_MAX_DELTA_MID = 0.28, 0.36
XT_V5_MAX_DELTA_ATT, XT_V5_MAX_DELTA_BOX = 0.42, 0.52
XT_V4_BOX_X_START = 90.0

RANKING_METRIC_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("All-around pass efficiency and impact", (
        "impact_passes_p90", "impact_per_pass",
        "phi_p90", "phi_per_pass",
    )),
    ("How much impact", ("dxt_p90", "dxt_per_pass")),
    ("How often impact", ("dxt_gt_01_pct",)),
    ("Construction & aggression efficiency", (
        "construction_aip", "construction_aip_per_pass",
        "aggression_aip", "aggression_aip_per_pass",
    )),
)

RATING_METRIC_KEYS: tuple[str, ...] = tuple(
    key for _, keys in RANKING_METRIC_GROUPS for key in keys
)

METRIC_LABELS: dict[str, str] = {
    "impact_passes_p90": "Passes Impact p90",
    "impact_per_pass": "Passes Impact / Pass",
    "phi_p90": "PHI p90",
    "phi_per_pass": "PHI / Pass",
    "dxt_p90": "ΔxT p90",
    "dxt_per_pass": "ΔxT / Pass",
    "dxt_gt_01_pct": "% passes ΔxT > 0.1",
    "construction_aip": "Construction AIP",
    "construction_aip_per_pass": "Construction AIP / Construction Passes",
    "aggression_aip": "Aggression AIP",
    "aggression_aip_per_pass": "Aggression AIP / Aggression Passes",
}

TOOLTIP_EXTRA_KEYS: tuple[str, ...] = ("minutes", "passes_completed")

ABSOLUTE_METRIC_KEYS: tuple[str, ...] = (
    "impact_passes_p90",
    "phi_p90",
    "dxt_p90",
    "construction_aip",
    "aggression_aip",
)

RELATIVE_METRIC_KEYS: tuple[str, ...] = (
    "impact_per_pass",
    "phi_per_pass",
    "dxt_per_pass",
    "dxt_gt_01_pct",
    "construction_aip_per_pass",
    "aggression_aip_per_pass",
)

LONG_BALL_STAT_KEYS: tuple[str, ...] = (
    "long_balls",
    "long_impact_passes",
)

SECTION_RATING_GROUPS: dict[str, tuple[str, ...]] = {
    "metrics_absolute": ABSOLUTE_METRIC_KEYS,
    "metrics_relative": RELATIVE_METRIC_KEYS,
    "long_balls": LONG_BALL_STAT_KEYS,
}

RANK_DISPLAY_KEYS: tuple[str, ...] = (
    *TOOLTIP_EXTRA_KEYS,
    "minutes_pct",
    *LONG_BALL_STAT_KEYS,
    *RATING_METRIC_KEYS,
)

TOOLTIP_LABELS: dict[str, str] = {
    **METRIC_LABELS,
    "minutes": "Minutos",
    "passes_completed": "Passes",
    "minutes_pct": "Min %",
    "long_balls": "Long balls (>30 m)",
    "long_impact_passes": "Long balls impact",
}


# ── Math helpers ──────────────────────────────────────────────────────────────
def _smoothstep(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _smootherstep(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0.0, 1.0)
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def _lateral_relative_position(y: np.ndarray) -> np.ndarray:
    return np.abs(y - GOAL_Y) / (FIELD_Y / 2.0)


def _gaussian_kernel_1d(sigma: float) -> np.ndarray:
    radius = max(1, int(np.ceil(3.0 * sigma)))
    xs = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (xs / sigma) ** 2)
    return kernel / kernel.sum()


def _gaussian_smooth_2d(grid: np.ndarray, sigma_x: float, sigma_y: float) -> np.ndarray:
    out = grid
    if sigma_x > 0:
        kx = _gaussian_kernel_1d(sigma_x)
        out = np.apply_along_axis(lambda row: np.convolve(row, kx, mode="same"), axis=1, arr=out)
    if sigma_y > 0:
        ky = _gaussian_kernel_1d(sigma_y)
        out = np.apply_along_axis(lambda row: np.convolve(row, ky, mode="same"), axis=0, arr=out)
    return out


def _map_zonal_threat_v31(x: np.ndarray) -> np.ndarray:
    blend = XT_V31_ZONE_BLEND_WIDTH
    x = np.clip(x, 0.0, FIELD_X)
    threat_def = XT_V3_DEF_MAX * _smootherstep(np.clip(x / OPT_ATTACKING_TWO_THIRDS_X, 0.0, 1.0))
    mid_span = max(FINAL_THIRD_LINE_X - OPT_ATTACKING_TWO_THIRDS_X, 1.0)
    mid_t = np.clip((x - OPT_ATTACKING_TWO_THIRDS_X) / mid_span, 0.0, 1.0)
    threat_mid = XT_V3_DEF_MAX + (XT_V3_MID_MAX - XT_V3_DEF_MAX) * _smootherstep(mid_t)
    att_span = max(FIELD_X - FINAL_THIRD_LINE_X, 1.0)
    att_t = np.clip((x - FINAL_THIRD_LINE_X) / att_span, 0.0, 1.0)
    threat_att = XT_V3_MID_MAX + (XT_V3_ATT_BYLINE - XT_V3_MID_MAX) * _smootherstep(att_t)
    w_def = 1.0 - _smootherstep(np.clip((x - (OPT_ATTACKING_TWO_THIRDS_X - blend)) / blend, 0.0, 1.0))
    w_att = _smootherstep(np.clip((x - (FINAL_THIRD_LINE_X - blend)) / blend, 0.0, 1.0))
    w_mid = np.clip(1.0 - w_def - w_att, 0.0, 1.0)
    w_sum = w_def + w_mid + w_att + 1e-12
    return (w_def * threat_def + w_mid * threat_mid + w_att * threat_att) / w_sum


def _location_factor_v31(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    lat = _lateral_relative_position(y)
    depth = np.clip((x - XT_V31_LAT_GATE_X) / (FIELD_X - XT_V31_LAT_GATE_X), 0.0, 1.0)
    zone_gate = _smootherstep(depth)
    max_discount = XT_V31_LAT_DISC_MAX * zone_gate
    lateral_curve = _smootherstep(lat ** XT_V3_LAT_CURVE_POWER)
    return 1.0 - max_discount * lateral_curve


def _build_heuristic_v31_threat_surface(Xc: np.ndarray, Yc: np.ndarray) -> np.ndarray:
    zonal = _map_zonal_threat_v31(Xc)
    surface = zonal * _location_factor_v31(Xc, Yc)
    surface = np.clip(surface, 0.0, XT_V3_SURFACE_MAX)
    return np.clip(_gaussian_smooth_2d(surface, XT_V31_GAUSS_SIGMA_X, XT_V31_GAUSS_SIGMA_Y), 0.0, XT_V3_SURFACE_MAX)


def _markov_quadrant_bonus_field(nx: int, ny: int) -> np.ndarray:
    from scipy.interpolate import RegularGridInterpolator

    grid = load_markov_model("top5").xT
    peak = max(float(grid.max()), 1e-9)
    rel = (grid / peak) ** XT_V4_MARKOV_BONUS_POWER
    bonus_coarse = rel * XT_V4_MARKOV_BONUS_MAX
    y_coords = np.linspace(0.0, FIELD_Y, grid.shape[0])
    x_coords = np.linspace(0.0, FIELD_X, grid.shape[1])
    interp = RegularGridInterpolator(
        (y_coords, x_coords), bonus_coarse, bounds_error=False, fill_value=0.0
    )
    xe = np.linspace(0.0, FIELD_X, nx)
    ye = np.linspace(0.0, FIELD_Y, ny)
    Xc, Yc = np.meshgrid(xe, ye)
    pts = np.column_stack([Yc.ravel(), Xc.ravel()])
    return interp(pts).reshape(ny, nx)


def _markov_final_third_envelope(Xc: np.ndarray) -> np.ndarray:
    t = _smootherstep(
        np.clip((Xc - (FINAL_THIRD_LINE_X - XT_V4_MARKOV_GATE_BLEND)) / XT_V4_MARKOV_GATE_BLEND, 0.0, 1.0)
    )
    return XT_V4_MARKOV_DEF_MID_FLOOR + (1.0 - XT_V4_MARKOV_DEF_MID_FLOOR) * t


def _build_heuristic_v4_fine_grid(nx: int = XT_V3_FINE_NX, ny: int = XT_V3_FINE_NY) -> np.ndarray:
    xe = np.linspace(0.0, FIELD_X, nx)
    ye = np.linspace(0.0, FIELD_Y, ny)
    Xc, Yc = np.meshgrid(xe, ye)
    base = _build_heuristic_v31_threat_surface(Xc, Yc)
    bonus = _markov_quadrant_bonus_field(nx, ny) * _markov_final_third_envelope(Xc)
    return np.clip(base + bonus, 0.0, XT_V4_SURFACE_MAX)


@functools.lru_cache(maxsize=1)
def _v4_interpolator():
    from scipy.interpolate import RegularGridInterpolator

    fine = _build_heuristic_v4_fine_grid()
    nx, ny = fine.shape[1], fine.shape[0]
    x_coords = np.linspace(0.0, FIELD_X, nx)
    y_coords = np.linspace(0.0, FIELD_Y, ny)
    return RegularGridInterpolator((y_coords, x_coords), fine, bounds_error=False, fill_value=0.0)


def _interp_xt(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    interp = _v4_interpolator()
    pts = np.column_stack([y, x])
    return interp(pts).astype(float)


def _short_pass_multiplier_vec(dist: np.ndarray) -> np.ndarray:
    blend_span = 4.0
    out = np.ones_like(dist, dtype=float)
    short = dist < XT_V4_SHORT_PASS_DIST
    blend = (dist >= XT_V4_SHORT_PASS_DIST) & (dist < XT_V4_SHORT_PASS_DIST + blend_span)
    out[short] = XT_V4_SHORT_PASS_FACTOR
    if blend.any():
        t = (dist[blend] - XT_V4_SHORT_PASS_DIST) / blend_span
        out[blend] = XT_V4_SHORT_PASS_FACTOR + (1.0 - XT_V4_SHORT_PASS_FACTOR) * t
    return out


def _zone_max_delta_vec(x_start: np.ndarray) -> np.ndarray:
    x = np.clip(x_start.astype(float), 0.0, FIELD_X)
    caps = np.full_like(x, XT_V5_MAX_DELTA_BOX)
    points = [
        (0.0, XT_V5_MAX_DELTA_DEF),
        (OPT_ATTACKING_TWO_THIRDS_X, XT_V5_MAX_DELTA_MID),
        (FINAL_THIRD_LINE_X, XT_V5_MAX_DELTA_ATT),
        (XT_V4_BOX_X_START, XT_V5_MAX_DELTA_BOX),
    ]
    for i in range(len(points) - 1):
        x0, c0 = points[i]
        x1, c1 = points[i + 1]
        mask = (x >= x0) & (x <= x1)
        if not mask.any():
            continue
        t = _smoothstep((x[mask] - x0) / max(x1 - x0, 1e-9))
        caps[mask] = c0 + (c1 - c0) * t
    return caps


def _adjust_delta_v4(
    is_won: np.ndarray,
    xt_start: np.ndarray,
    xt_end: np.ndarray,
    x_start: np.ndarray,
    y_start: np.ndarray,
    x_end: np.ndarray,
    y_end: np.ndarray,
    pass_distance: np.ndarray,
) -> np.ndarray:
    raw = np.where(is_won, xt_end - xt_start, 0.0)
    mult = _short_pass_multiplier_vec(pass_distance)
    pos = raw >= 0
    adjusted = np.where(pos, np.minimum(raw * mult, _zone_max_delta_vec(x_start)), raw)

    lat_start = np.abs(y_start - GOAL_Y) / (FIELD_Y / 2.0)
    lat_end = np.abs(y_end - GOAL_Y) / (FIELD_Y / 2.0)
    neg_recycle = (~pos) & (x_start < XT_V3_NEG_RECYCLE_X_MAX)
    adjusted = np.where(neg_recycle & (lat_end < lat_start), raw * XT_V3_NEG_PENALTY_FACTOR, adjusted)
    pressure = (
        (~pos)
        & (x_start < XT_V3_PRESSURE_X_MAX)
        & (lat_start > XT_V3_WIDE_FRAC)
        & (lat_end < lat_start - 0.12)
    )
    adjusted = np.where(pressure, adjusted + XT_V3_PRESSURE_ESCAPE_BONUS, adjusted)
    return adjusted


def _impact_tier_vec(xt_start: np.ndarray, delta_xt: np.ndarray) -> np.ndarray:
    """0 = none, 1 = progressive, 2 = highly."""
    tier = np.zeros(len(delta_xt), dtype=np.int8)
    pos = delta_xt > 0
    if not pos.any():
        return tier
    prog = np.maximum(XT_V3_PROG_FLOOR_CLASS, XT_V3_PROG_SCALE_CLASS * (1.0 - xt_start)) * IMPACT_PROG_STRICTNESS
    high = np.maximum(XT_V3_HIGH_FLOOR_CLASS, XT_V3_HIGH_SCALE_CLASS * (1.0 - xt_start))
    tier[pos & (delta_xt > prog) & (delta_xt <= high)] = 1
    tier[pos & (delta_xt > high)] = 2
    return tier


def _goal_dist_vec(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.sqrt((GOAL_X - x) ** 2 + (GOAL_Y - y) ** 2)


def _approaches_goal_vec(
    x_start: np.ndarray, y_start: np.ndarray, x_end: np.ndarray, y_end: np.ndarray
) -> np.ndarray:
    progress = _goal_dist_vec(x_start, y_start) - _goal_dist_vec(x_end, y_end)
    min_app = np.where(x_end >= FINAL_THIRD_LINE_X, IMPACT_PASS_MIN_GOAL_APPROACH_FINAL_THIRD, IMPACT_PASS_MIN_GOAL_APPROACH_REST)
    return progress >= min_app


def _progressive_wyscout_vec(
    x_start: np.ndarray, y_start: np.ndarray, x_end: np.ndarray, y_end: np.ndarray
) -> np.ndarray:
    progress = _goal_dist_vec(x_start, y_start) - _goal_dist_vec(x_end, y_end)
    ok = progress > 0
    out = np.zeros(len(progress), dtype=bool)
    if not ok.any():
        return out
    start_own = x_start < HALF_LINE_X
    end_own = x_end < HALF_LINE_X
    start_opp = x_start >= HALF_LINE_X
    end_opp = x_end >= HALF_LINE_X
    thresh = np.full(len(progress), WYSCOUT_PROG_CROSS_HALF)
    thresh[start_own & end_own] = WYSCOUT_PROG_OWN_HALF
    thresh[start_opp & end_opp] = WYSCOUT_PROG_OPP_HALF
    out[ok] = progress[ok] >= thresh[ok]
    return out


# ── Data loading ──────────────────────────────────────────────────────────────
def _parse_bool_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "successful"})


def _wyscout_to_sb(x: pd.Series, y: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    x_sb = x.to_numpy(dtype=float) * FIELD_X / WYSCOUT_PITCH_SIZE
    y_sb = FIELD_Y - (y.to_numpy(dtype=float) * FIELD_Y / WYSCOUT_PITCH_SIZE)
    return x_sb, y_sb


def _normalize_position(raw: str | None) -> str:
    return normalize_sofascore_position(raw, default=DEFAULT_PLAYER_POSITION)


def build_player_registry(frame: pd.DataFrame) -> list[dict]:
    work = frame.copy()
    work["player_id"] = work["player_id"].astype(str)
    if "position" in work.columns:
        work["position"] = work["position"].map(_normalize_position)
        pos_by_id = (
            work.groupby("player_id", sort=False)["position"]
            .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else DEFAULT_PLAYER_POSITION)
            .to_dict()
        )
    else:
        pos_by_id = {}
    players_df = work[["player_id", "player_name"]].drop_duplicates().sort_values("player_name")
    return [
        {
            "code": str(row.player_id),
            "name": str(row.player_name),
            "position": pos_by_id.get(str(row.player_id), DEFAULT_PLAYER_POSITION),
        }
        for row in players_df.itertuples(index=False)
    ]


def _enrich_passes(frame: pd.DataFrame) -> pd.DataFrame:
    sx, sy = _wyscout_to_sb(frame["start_x"], frame["start_y"])
    has_end = frame["end_x"].notna() & frame["end_y"].notna()
    ex = np.full(len(frame), np.nan)
    ey = np.full(len(frame), np.nan)
    if has_end.any():
        ex[has_end.to_numpy()], ey[has_end.to_numpy()] = _wyscout_to_sb(
            frame.loc[has_end, "end_x"], frame.loc[has_end, "end_y"]
        )

    out = pd.DataFrame({
        "player_id": frame["player_id"].astype(str),
        "player_name": frame["player_name"].astype(str),
        "position": frame["position"].map(_normalize_position) if "position" in frame.columns else DEFAULT_PLAYER_POSITION,
        "is_success": _parse_bool_series(frame["outcome"]) if "outcome" in frame.columns else False,
        "is_key_pass": _parse_bool_series(frame["keypass"]) if "keypass" in frame.columns else False,
        "action_type": frame["eventActionType"].astype(str).str.strip().str.lower(),
        "x_start": sx,
        "y_start": sy,
        "x_end": ex,
        "y_end": ey,
        "has_end": has_end.to_numpy(),
    })
    out["is_won"] = out["is_success"].astype(bool)
    out["pass_distance"] = np.where(
        out["has_end"],
        np.sqrt((out["x_end"] - out["x_start"]) ** 2 + (out["y_end"] - out["y_start"]) ** 2),
        0.0,
    )
    out["is_long_ball"] = out["has_end"] & (out["pass_distance"] >= LONG_PASS_MIN_DISTANCE_M)

    mask = out["has_end"].to_numpy()
    if mask.any():
        sub = out.loc[mask]
        xt_start = _interp_xt(sub["x_start"].to_numpy(), sub["y_start"].to_numpy())
        xt_end = _interp_xt(sub["x_end"].to_numpy(), sub["y_end"].to_numpy())
        delta = _adjust_delta_v4(
            sub["is_won"].to_numpy(),
            xt_start, xt_end,
            sub["x_start"].to_numpy(), sub["y_start"].to_numpy(),
            sub["x_end"].to_numpy(), sub["y_end"].to_numpy(),
            sub["pass_distance"].to_numpy(),
        )
        out.loc[mask, "xt_start_v4"] = xt_start
        out.loc[mask, "xt_end_v4"] = xt_end
        out.loc[mask, "delta_xt_v4"] = delta
    else:
        out["xt_start_v4"] = 0.0
        out["xt_end_v4"] = 0.0
        out["delta_xt_v4"] = 0.0

    tier = _impact_tier_vec(out["xt_start_v4"].to_numpy(), out["delta_xt_v4"].to_numpy())
    approaches = _approaches_goal_vec(
        out["x_start"].to_numpy(), out["y_start"].to_numpy(),
        out["x_end"].to_numpy(), out["y_end"].to_numpy(),
    )
    out["impact_tier"] = tier
    out["approaches_goal"] = approaches
    out["is_progressive_wyscout"] = _progressive_wyscout_vec(
        out["x_start"].to_numpy(), out["y_start"].to_numpy(),
        out["x_end"].to_numpy(), out["y_end"].to_numpy(),
    )
    out["impact_attempt"] = out["has_end"] & out["approaches_goal"] & (out["impact_tier"] >= 1)
    out["high_impact_attempt"] = out["has_end"] & out["approaches_goal"] & (out["impact_tier"] >= 2)
    out["impact_success"] = out["is_won"] & out["impact_attempt"]
    out["high_impact_success"] = out["is_won"] & out["high_impact_attempt"]
    out["prog_success"] = out["is_success"] & out["is_progressive_wyscout"]
    return out


def _minutes_from_passes_frame(frame: pd.DataFrame) -> dict[str, dict]:
    """Derive team, minutes estimate and % from pass events (Série B)."""
    work = frame.copy()
    work["player_id"] = work["player_id"].astype(str)
    is_home = _parse_bool_series(work["isHome"])
    work["team"] = np.where(is_home, work["home_team"], work["away_team"])
    team_games = work.groupby("team", sort=False)["event_id"].nunique().to_dict()

    out: dict[str, dict] = {}
    for pid, grp in work.groupby("player_id", sort=False):
        team = str(grp["team"].mode().iloc[0] if not grp["team"].mode().empty else grp["team"].iloc[0])
        games = int(grp["event_id"].nunique())
        max_games = int(team_games.get(team, games))
        pct = games / max_games if max_games > 0 else None
        out[pid] = {
            "team": team,
            "minutes": games * 90,
            "minutes_pct": round(pct, 4) if pct is not None else None,
            "eligible_ranking": pct is not None and pct > RATING_MIN_MINUTES_PCT,
        }
    return out


@functools.lru_cache(maxsize=1)
def _load_minutes_info_sofa() -> dict[str, dict]:
    if not PLAYER_MATCH_STATS_PATH.exists():
        return {}
    stats = pd.read_csv(PLAYER_MATCH_STATS_PATH, low_memory=False)
    if stats.empty or "player_id" not in stats.columns:
        return {}
    stats["player_id"] = stats["player_id"].astype(str)
    stats["minutes_played"] = pd.to_numeric(stats.get("minutes_played", 0), errors="coerce").fillna(0.0)
    is_home = stats["is_home"].astype(str).str.strip().str.lower().isin({"true", "1", "yes"})
    stats["team"] = np.where(is_home, stats["home_team"], stats["away_team"])
    team_matches = stats.groupby("team")["event_id"].nunique().to_dict() if "event_id" in stats.columns else {}

    out: dict[str, dict] = {}
    for pid, grp in stats.groupby("player_id", sort=False):
        minutes = float(grp["minutes_played"].sum())
        team = str(grp["team"].mode().iloc[0] if not grp["team"].mode().empty else grp["team"].iloc[0])
        max_minutes = float(team_matches.get(team, 0) * 90)
        pct = (minutes / max_minutes) if max_minutes > 0 else None
        out[pid] = {
            "team": team,
            "minutes": int(round(minutes)),
            "minutes_pct": round(pct, 4) if pct is not None else None,
            "eligible_ranking": pct is not None and pct >= MIN_MINUTES_PCT,
        }
    return out


def _load_minutes_info(frame: pd.DataFrame) -> dict[str, dict]:
    """Prefer SofaScore minutes when available; otherwise derive from pass events."""
    derived = _minutes_from_passes_frame(frame)
    sofa = _load_minutes_info_sofa()
    if not sofa:
        return derived
    merged = dict(derived)
    merged.update(sofa)
    return merged


def _accuracy(attempt: pd.Series, success: pd.Series) -> dict:
    attempted = int(attempt.sum())
    successful = int((attempt & success).sum())
    return {
        "successful": successful,
        "attempted": attempted,
        "accuracy_pct": round(successful / attempted * 100.0, 1) if attempted else 0.0,
    }


def _safe_ratio(num: float, den: int, *, decimals: int = 3) -> float:
    return round(float(num) / den, decimals) if den else 0.0


def _per90(total: float, minutes: float | None) -> float:
    return round(float(total) * 90.0 / float(minutes), 3) if minutes else 0.0


def _zone_metrics(passes: pd.DataFrame, construction: bool) -> dict:
    if construction:
        mask = passes["has_end"] & (passes["x_end"] < PASS_AGGRESSION_X_MIN)
    else:
        mask = passes["has_end"] & (passes["x_end"] >= PASS_AGGRESSION_X_MIN)
    zone = passes[mask]
    completed = zone[zone["is_success"]]
    return {
        "passes": int(len(completed)),
        "progressive_passes": int(zone["prog_success"].sum()),
        "impact_passes": int(zone["impact_success"].sum()),
        "high_impact_passes": int(zone["high_impact_success"].sum()),
        "sum_dxt": float(zone["delta_xt_v4"].sum()),
        "sum_xt_end": float(completed["xt_end_v4"].sum()) if not completed.empty else 0.0,
    }


def _pass_layer_metrics(passes: pd.DataFrame) -> dict:
    if passes.empty:
        return {}
    completed = passes[passes["is_success"]]
    total = len(passes)
    xt = passes[passes["has_end"]]
    impact = _accuracy(passes["impact_attempt"], passes["impact_success"])
    high = _accuracy(passes["high_impact_attempt"], passes["high_impact_success"])

    dxt_gt_01_pct = float((xt["delta_xt_v4"] > DXT_IMPACT_THRESHOLD).mean() * 100.0) if len(xt) else 0.0

    construction = _zone_metrics(passes, True)
    aggression = _zone_metrics(passes, False)
    construction_aip = construction["impact_passes"] + construction["high_impact_passes"]
    aggression_aip = aggression["impact_passes"] + aggression["high_impact_passes"]

    return {
        "passes_total": total,
        "passes_completed": int(len(completed)),
        "impact_passes": impact["successful"],
        "impact_attempted": impact["attempted"],
        "impact_accuracy_pct": impact["accuracy_pct"],
        "high_impact_passes": high["successful"],
        "high_impact_attempted": high["attempted"],
        "high_impact_accuracy_pct": high["accuracy_pct"],
        "sum_dxt_passes": float(passes["delta_xt_v4"].sum()),
        "sum_xt_end_passes": float(completed["xt_end_v4"].sum()) if not completed.empty else 0.0,
        "dxt_gt_01_pct": round(dxt_gt_01_pct, 1),
        "impact_per_pass": _safe_ratio(impact["successful"], total),
        "phi_per_pass": _safe_ratio(high["successful"], total),
        "dxt_per_pass": _safe_ratio(float(passes["delta_xt_v4"].sum()), int(len(completed))),
        "construction_aip": int(construction_aip),
        "construction_aip_per_pass": _safe_ratio(construction_aip, construction["passes"]),
        "aggression_aip": int(aggression_aip),
        "aggression_aip_per_pass": _safe_ratio(aggression_aip, aggression["passes"]),
        "construction_passes": construction["passes"],
        "aggression_passes": aggression["passes"],
    }


def _derive_rates(stats: dict, minutes: float | None) -> dict:
    out = dict(stats)
    out["impact_passes_p90"] = _per90(stats.get("impact_passes", 0), minutes)
    out["phi_p90"] = _per90(stats.get("high_impact_passes", 0), minutes)
    out["dxt_p90"] = _per90(stats.get("sum_dxt_passes", 0), minutes)
    return out


def _long_pass_mask(passes: pd.DataFrame) -> pd.Series:
    """Passes com destino e distância >= 30 m (StatsBomb, metros)."""
    if passes.empty:
        return pd.Series(dtype=bool)
    has_end = passes["has_end"].fillna(False).astype(bool)
    dist = np.sqrt(
        (passes["x_end"].to_numpy(dtype=float) - passes["x_start"].to_numpy(dtype=float)) ** 2
        + (passes["y_end"].to_numpy(dtype=float) - passes["y_start"].to_numpy(dtype=float)) ** 2
    )
    return has_end & (dist >= LONG_PASS_MIN_DISTANCE_M)


def _long_ball_stats(passes: pd.DataFrame) -> dict:
    mask = _long_pass_mask(passes)
    long_passes = passes[mask]
    n_long = int(mask.sum())
    if n_long == 0:
        return {
            "long_balls": 0,
            "long_balls_completed": 0,
            "long_impact_passes": 0,
            "long_impact_eff_pct": 0.0,
            "long_impact_per_long_pass": 0.0,
        }
    layer = _pass_layer_metrics(long_passes)
    n_impact = int(layer.get("impact_passes", 0))
    return {
        "long_balls": n_long,
        "long_balls_completed": int(long_passes["is_success"].sum()),
        "long_impact_passes": n_impact,
        "long_impact_eff_pct": float(layer.get("impact_accuracy_pct", 0.0)),
        "long_impact_per_long_pass": _safe_ratio(n_impact, n_long),
    }


def compute_player_metrics(passes: pd.DataFrame, minutes_info: dict) -> dict:
    stats = {**_pass_layer_metrics(passes), **_long_ball_stats(passes)}
    minutes = minutes_info.get("minutes")
    return _derive_rates(stats, minutes)


def _rank_to_rating_score(rank: int, pool_size: int) -> float:
    if pool_size <= 1:
        return RATING_SCORE_BEST
    span = RATING_SCORE_BEST - RATING_SCORE_WORST
    return RATING_SCORE_WORST + span * (pool_size - rank) / (pool_size - 1)


def _metric_ranks_for_pool(pool: list[dict]) -> dict[str, dict[str, dict]]:
    """player_id -> metric_key -> {rank, total, value}."""
    n = len(pool)
    if n == 0:
        return {}
    keys = list(RANK_DISPLAY_KEYS)
    out: dict[str, dict[str, dict]] = {p["player_id"]: {} for p in pool}
    for key in keys:
        ordered = sorted(pool, key=lambda p: p.get(key, 0) or 0, reverse=True)
        for rank, player in enumerate(ordered, start=1):
            out[player["player_id"]][key] = {
                "rank": rank,
                "total": n,
                "value": player.get(key),
            }
    return out


def _section_ratings_for_pool(pos_players: list[dict], pool_size: int) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for section_key, keys in SECTION_RATING_GROUPS.items():
        scores: dict[str, list[float]] = {p["player_id"]: [] for p in pos_players}
        for key in keys:
            ordered = sorted(pos_players, key=lambda p: p.get(key, 0) or 0, reverse=True)
            for rank, player in enumerate(ordered, start=1):
                scores[player["player_id"]].append(_rank_to_rating_score(rank, pool_size))
        out[section_key] = {
            pid: round(sum(vals) / len(vals), 4) if vals else 0.0
            for pid, vals in scores.items()
        }
    return out


def _section_rating_ranks_for_pool(section_scores: dict[str, dict[str, float]], pool_size: int) -> dict[str, dict[str, dict]]:
    """section_key -> player_id -> {rank, total, value}."""
    ranks: dict[str, dict[str, dict]] = {}
    for section_key, by_player in section_scores.items():
        ordered = sorted(by_player.items(), key=lambda item: item[1], reverse=True)
        ranks[section_key] = {}
        for rank, (pid, value) in enumerate(ordered, start=1):
            ranks[section_key][pid] = {"rank": rank, "total": pool_size, "value": value}
    return ranks


def compute_pass_ratings(players: list[dict]) -> list[dict]:
    by_position: dict[str, list[dict]] = {}
    for player in players:
        by_position.setdefault(str(player.get("position") or "—"), []).append(player)
    rated: list[dict] = []
    for pos, pos_players in by_position.items():
        pool_size = len(pos_players)
        if pool_size == 0:
            continue
        metric_ranks = _metric_ranks_for_pool(pos_players)
        section_scores = _section_ratings_for_pool(pos_players, pool_size)
        section_rating_ranks = _section_rating_ranks_for_pool(section_scores, pool_size)
        scores: dict[str, list[float]] = {p["player_id"]: [] for p in pos_players}
        for key in RATING_METRIC_KEYS:
            ordered = sorted(pos_players, key=lambda p: p.get(key, 0) or 0, reverse=True)
            for rank, player in enumerate(ordered, start=1):
                scores[player["player_id"]].append(_rank_to_rating_score(rank, pool_size))
        pool_entries: list[dict] = []
        for player in pos_players:
            vals = scores[player["player_id"]]
            pass_rating = round(sum(vals) / len(vals), 4) if vals else 0.0
            pool_entries.append({
                **player,
                "pass_rating": pass_rating,
                "metric_ranks": dict(metric_ranks.get(player["player_id"], {})),
                "section_ratings": {
                    sk: section_scores[sk].get(player["player_id"], 0.0)
                    for sk in SECTION_RATING_GROUPS
                },
                "section_rating_ranks": {
                    sk: section_rating_ranks[sk].get(player["player_id"], {})
                    for sk in SECTION_RATING_GROUPS
                },
            })
        pool_entries.sort(key=lambda p: p.get("pass_rating", 0), reverse=True)
        for rank, player in enumerate(pool_entries, start=1):
            player["metric_ranks"]["pass_rating"] = {
                "rank": rank,
                "total": pool_size,
                "value": player["pass_rating"],
            }
        rated.extend(pool_entries)
    return rated


@functools.lru_cache(maxsize=1)
def load_passes_grouped(cache_version: int = DATA_CACHE_VERSION) -> dict[str, pd.DataFrame]:
    """Enriched passes indexed by player_id (for impact maps)."""
    _ = cache_version
    if not SEASON_ALL_CSV_PATH.exists():
        return {}
    frame = pd.read_csv(SEASON_ALL_CSV_PATH, low_memory=False)
    frame = frame[frame["category"].astype(str).str.lower() == "passes"]
    passes = _enrich_passes(frame)
    return {str(pid): grp for pid, grp in passes.groupby("player_id", sort=False)}


def build_analytics(cache_version: int = DATA_CACHE_VERSION) -> tuple[list[dict], list[dict]]:
    """Load CSV once, compute all player metrics. Returns (registry, eligible_players)."""
    _ = cache_version
    if not SEASON_ALL_CSV_PATH.exists():
        return [], []

    frame = pd.read_csv(SEASON_ALL_CSV_PATH, low_memory=False)
    frame = frame[frame["category"].astype(str).str.lower() == "passes"]
    registry = build_player_registry(frame)
    passes = _enrich_passes(frame)
    minutes_info = _load_minutes_info(frame)

    players: list[dict] = []
    for player in registry:
        if not is_outfield_position(player.get("position")):
            continue
        pid = player["code"]
        mins = minutes_info.get(pid, {})
        pct = mins.get("minutes_pct")
        if pct is None or pct <= RATING_MIN_MINUTES_PCT:
            continue
        grp = passes[passes["player_id"] == pid]
        if grp.empty:
            continue
        metrics = compute_player_metrics(grp, mins)
        players.append({
            "player_id": pid,
            "player_name": player["name"],
            "position": player.get("position", "—"),
            "position_group": position_group(player.get("position")),
            "team": mins.get("team", "—"),
            "minutes": mins.get("minutes"),
            "minutes_pct": pct,
            **{k: round(v, 4) if isinstance(v, float) and abs(v) < 1000 else v for k, v in metrics.items()},
        })
    return registry, players


def metric_label(key: str) -> str:
    return TOOLTIP_LABELS.get(key, key.replace("_", " ").title())


def fmt_smart(value, *, max_decimals: int = 4) -> str:
    """Adaptive decimals: extend when 1 dp rounds to 0.0 on a non-zero value."""
    if value is None:
        return "—"
    v = float(value)
    if v == 0.0:
        return "0.0"
    if abs(v - round(v)) < 1e-9 and abs(v) >= 1.0:
        return str(int(round(v)))
    for decimals in range(1, max_decimals + 1):
        text = f"{v:.{decimals}f}"
        if decimals == max_decimals or float(text) != 0.0:
            return text
    return f"{v:.{max_decimals}f}"


def fmt_stat_value(key: str, value) -> str:
    if value is None:
        return "—"
    if key.endswith("_pct"):
        return f"{fmt_smart(value)}%"
    if key in {
        "minutes", "passes_completed", "long_balls", "long_balls_completed",
        "long_impact_passes", "impact_passes", "high_impact_passes",
        "construction_aip", "aggression_aip", "construction_passes", "aggression_passes",
    }:
        return fmt_smart(value, max_decimals=1) if float(value) == int(float(value)) else fmt_smart(value)
    if "per_" in key or key.endswith("_p90"):
        return fmt_smart(value)
    if isinstance(value, float):
        return fmt_smart(value)
    return fmt_smart(value) if isinstance(value, (int, float)) else str(value)


def fmt_metric_value(key: str, value) -> str:
    return fmt_stat_value(key, value)


def fmt_count(value) -> str:
    return fmt_smart(value, max_decimals=1)


def fmt_pct(value: float) -> str:
    return f"{fmt_smart(value)}%"


def fmt_rating_score(pass_rating) -> str:
    if pass_rating is None:
        return "—"
    return f"{float(pass_rating) * 10.0:.2f}"


def fmt_decimal(value, *, decimals: int = 3) -> str:
    if value is None:
        return "—"
    return fmt_smart(value, max_decimals=decimals)
