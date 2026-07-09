"""Passes xTh — Série B: rating por posição e mapa de passes de impacto."""

from __future__ import annotations

import html
import inspect
import sys
import unicodedata
from pathlib import Path

_APP_ROOT = Path(__file__).resolve().parent
for _path in (_APP_ROOT, _APP_ROOT / "scripts"):
    _entry = str(_path)
    if _entry not in sys.path:
        sys.path.insert(0, _entry)


def _load_similarity_engine():
    """Load local similarity_engine.py explicitly (avoids path/shadowing on Streamlit Cloud)."""
    import importlib.util

    module_path = _APP_ROOT / "similarity_engine.py"
    if not module_path.is_file():
        raise ImportError(f"Arquivo não encontrado: {module_path}")
    spec = importlib.util.spec_from_file_location("passes_xt_similarity_engine", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Não foi possível carregar {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules["passes_xt_similarity_engine"] = module
    return module

import streamlit as st
import streamlit.components.v1 as components

import passes_engine as pe
sim = _load_similarity_engine()
from comparison_config import (
    CLASSIFICATION_MODEL_DEFAULT,
    TIER_MODEL_DEFAULT,
    XT_SURFACE_MODE_DEFAULT,
    normalize_classification_model,
    normalize_tier_model,
    normalize_xt_surface_mode,
)
from passes_maps import (
    draw_all_completed_passes_map,
    draw_impact_pass_map,
    draw_pass_destination_heatmap,
    draw_pass_origin_heatmap,
)

DATA_CACHE_VERSION = pe.DATA_CACHE_VERSION
LONG_BALL_STAT_KEYS = pe.LONG_BALL_STAT_KEYS
ABSOLUTE_METRIC_KEYS = pe.ABSOLUTE_METRIC_KEYS
RELATIVE_METRIC_KEYS = pe.RELATIVE_METRIC_KEYS
CONSTRUCTION_METRIC_KEYS = pe.CONSTRUCTION_METRIC_KEYS
AGGRESSION_METRIC_KEYS = pe.AGGRESSION_METRIC_KEYS
SCOUT_SECTION_SPECS = pe.SCOUT_SECTION_SPECS
POSITION_GROUPS_ORDER = pe.POSITION_GROUPS_ORDER
RATING_TOP_N = pe.RATING_TOP_N
RATING_MIN_MINUTES_PCT = pe.RATING_MIN_MINUTES_PCT
RATING_MIN_PASSES_PCT = pe.RATING_MIN_PASSES_PCT
SIMILARITY_TOP_K = 10
SIMILARITY_SELECT_SB_KEY = "similarity_player_select_sb"
SIMILARITY_SELECT_SA_KEY = "similarity_player_select_sa"
FIXED_CLASSIFICATION_MODEL = CLASSIFICATION_MODEL_DEFAULT
FIXED_TIER_MODEL = TIER_MODEL_DEFAULT
FIXED_XT_SURFACE_MODE = XT_SURFACE_MODE_DEFAULT
build_analytics = pe.build_analytics
compute_pass_ratings = pe.compute_pass_ratings
fmt_pct = pe.fmt_pct
fmt_stat_value = pe.fmt_stat_value
load_passes_grouped = pe.load_passes_grouped
metric_label = pe.metric_label
analyst_metric_label = pe.analyst_metric_label
metric_tooltip = pe.metric_tooltip
rank_in_group_label = pe.rank_in_group_label
rank_to_display_score = pe.rank_to_display_score
score_display_color = pe.score_display_color
rate_player_vs_eligible_pool = pe.rate_player_vs_eligible_pool
enrich_player_eligibility = pe.enrich_player_eligibility


def fmt_rating_score(pass_rating) -> str:
    if pass_rating is None:
        return "—"
    return f"{float(pass_rating) * 10.0:.1f}"

st.set_page_config(page_title="Passes xTh", layout="wide", initial_sidebar_state="collapsed")

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.25rem; max-width: 1600px; }
    .player-card {
        background: linear-gradient(160deg, #151b2b 0%, #101522 100%);
        border: 1px solid #2a3550;
        border-radius: 12px;
        padding: 1rem 1.1rem;
        margin-bottom: 0.65rem;
    }
    .player-info-card .player-header-stats {
        display: grid;
        grid-template-columns: 1fr;
        gap: 0.5rem;
        justify-content: stretch;
        margin-top: 0.75rem;
    }
    .player-info-card .rating-row { margin-top: 0.75rem; }
    .player-info-card .header-stat strong { font-size: 0.98rem; }
    .header-stat {
        font-size: 0.84rem;
        color: #94a3b8;
        white-space: nowrap;
    }
    .header-stat strong {
        display: block;
        color: #f8fafc;
        font-size: 1.02rem;
        font-weight: 700;
        margin-top: 0.1rem;
    }
    .rating-row {
        display: flex;
        align-items: center;
        flex-wrap: wrap;
        gap: 0.55rem;
        margin-bottom: 0;
    }
    .rating-warning-tip {
        position: relative;
        display: inline-flex;
        align-items: center;
    }
    .rating-warning {
        font-size: 1.2rem;
        line-height: 1;
        cursor: help;
        color: #fbbf24;
        filter: drop-shadow(0 0 4px rgba(251, 191, 36, 0.35));
    }
    .player-card h3 { margin: 0 0 0.15rem 0; color: #f1f5f9; font-size: 1.15rem; }
    .player-card .sub { color: #94a3b8; font-size: 0.85rem; margin-bottom: 0; }
    .player-card .rating-box {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        min-width: 76px;
        height: 50px;
        padding: 0 12px;
        border-radius: 8px;
        font-size: 1.55rem;
        font-weight: 800;
        margin-bottom: 0;
        border: 1px solid rgba(255,255,255,0.16);
        letter-spacing: 0.02em;
    }
    .metric-line .stat-val {
        font-size: 1.05rem;
        font-weight: 700;
        color: #f8fafc;
    }
    .metric-line {
        display: flex;
        justify-content: space-between;
        gap: 0.75rem;
        padding: 0.32rem 0;
        border-bottom: 1px solid #1f293f;
        font-size: 0.88rem;
        color: #cbd5e1;
    }
    .metric-line span:last-child { white-space: nowrap; }
    .val-wrap { display: inline-flex; align-items: center; gap: 0.5rem; }
    .rank-badge {
        display: inline-block;
        width: 12px;
        height: 12px;
        min-width: 12px;
        border-radius: 3px;
        flex-shrink: 0;
        border: 1px solid rgba(255,255,255,0.2);
        cursor: help;
    }
    .rank-tip, .rating-tip, .section-rating-tip {
        position: relative;
        display: inline-flex;
    }
    .rank-tipbox, .rating-tipbox {
        display: none;
        position: absolute;
        z-index: 100;
        left: 50%;
        bottom: calc(100% + 6px);
        transform: translateX(-50%);
        background: #111827;
        border: 1px solid #3d4f6f;
        border-radius: 6px;
        padding: 4px 8px;
        font-size: 0.72rem;
        font-weight: 700;
        color: #e2e8f0;
        white-space: nowrap;
        box-shadow: 0 8px 20px rgba(0,0,0,.4);
        pointer-events: none;
    }
    .rank-tip:hover .rank-tipbox,
    .rating-tip:hover .rating-tipbox,
    .section-rating-tip:hover .rating-tipbox,
    .rating-warning-tip:hover .rating-tipbox,
    .metric-tip:hover .metric-tipbox {
        display: block;
    }
    .metric-tip {
        position: relative;
        display: inline-flex;
        align-items: center;
        cursor: help;
        border-bottom: 1px dotted #475569;
    }
    .metric-tipbox {
        display: none;
        position: absolute;
        z-index: 120;
        left: 0;
        bottom: calc(100% + 6px);
        min-width: 200px;
        max-width: 280px;
        background: #111827;
        border: 1px solid #3d4f6f;
        border-radius: 8px;
        padding: 8px 10px;
        font-size: 0.72rem;
        font-weight: 500;
        line-height: 1.35;
        color: #e2e8f0;
        white-space: normal;
        box-shadow: 0 8px 20px rgba(0,0,0,.45);
        pointer-events: none;
    }
    .metric-rank-sub {
        display: block;
        margin-top: 0.12rem;
        font-size: 0.72rem;
        font-weight: 500;
        color: #64748b;
        letter-spacing: 0.01em;
    }
    .cmp-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.65rem 1.25rem;
        margin-top: 0.5rem;
    }
    .cmp-section-title {
        grid-column: 1 / -1;
        color: #93c5fd;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-top: 0.35rem;
        padding-top: 0.35rem;
        border-top: 1px solid #1f293f;
    }
    .cmp-section-title:first-child { border-top: none; margin-top: 0; padding-top: 0; }
    .cmp-row {
        display: grid;
        grid-template-columns: 1.1fr 1fr 1fr;
        gap: 0.75rem;
        align-items: end;
        padding: 0.45rem 0;
        border-bottom: 1px solid #1a2236;
    }
    .cmp-row-head {
        color: #94a3b8;
        font-size: 0.74rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        padding-bottom: 0.2rem;
        border-bottom: 1px solid #2a3550;
    }
    .cmp-cell-label { color: #cbd5e1; font-size: 0.84rem; }
    .cmp-cell-value {
        font-size: 1.05rem;
        font-weight: 700;
        color: #f8fafc;
    }
    .pres-card {
        background: linear-gradient(160deg, #151b2b 0%, #101522 100%);
        border: 1px solid #2a3550;
        border-radius: 12px;
        padding: 1rem 1.15rem;
        margin-bottom: 0.85rem;
    }
    .pres-card h4 { margin: 0 0 0.35rem 0; color: #e2e8f0; font-size: 1rem; }
    .pres-card p { margin: 0; color: #94a3b8; font-size: 0.88rem; line-height: 1.45; }
    .pres-step {
        display: flex;
        gap: 0.75rem;
        align-items: flex-start;
        margin: 0.55rem 0;
    }
    .pres-step-num {
        flex-shrink: 0;
        width: 1.55rem;
        height: 1.55rem;
        border-radius: 999px;
        background: #1e3a8a;
        color: #dbeafe;
        font-size: 0.78rem;
        font-weight: 800;
        display: inline-flex;
        align-items: center;
        justify-content: center;
    }
    .grade-card {
        background: linear-gradient(160deg, #151b2b 0%, #101522 100%);
        border: 1px solid #2a3550;
        border-radius: 10px;
        padding: 0.85rem 0.9rem;
        min-height: 112px;
        margin-bottom: 0.35rem;
    }
    .grade-card-title {
        color: #93c5fd;
        font-size: 0.74rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        line-height: 1.25;
    }
    .grade-card-sub {
        color: #64748b;
        font-size: 0.72rem;
        line-height: 1.35;
        margin-top: 0.3rem;
    }
    .grade-card-score { margin-top: 0.55rem; }
    .grade-card-rank {
        margin-top: 0.28rem;
        font-size: 0.72rem;
        color: #64748b;
    }
    .cmp-delta {
        display: inline-block;
        font-size: 0.58rem;
        line-height: 1;
        margin-left: 0.3rem;
        vertical-align: middle;
        font-weight: 800;
    }
    .cmp-delta.up { color: #34d399; }
    .cmp-delta.down { color: #f87171; }
    .cmp-delta.flat { color: #475569; }
    .cmp-value-wrap { display: inline-flex; align-items: center; }
    .stat-section-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 0.6rem;
        margin-top: 0.7rem;
        margin-bottom: 0.25rem;
    }
    .stat-section {
        color: #93c5fd;
        font-size: 0.74rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }
    .section-rating-pill {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        min-width: 52px;
        padding: 4px 11px;
        border-radius: 7px;
        font-size: 0.82rem;
        font-weight: 800;
        letter-spacing: 0.02em;
        border: 1px solid rgba(255,255,255,0.18);
        white-space: nowrap;
    }
    section[data-testid="stSidebar"] { display: none; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Passes xTh — Série B")

RATING_COLUMNS = ["Jogador", "Time", "Rating"]
SELECTBOX_KEY = "map_player_select"


def _call_build_analytics(
    cache_version: int,
    tier_model: str,
    classification_model: str,
    xt_surface_mode: str,
):
    sig = inspect.signature(build_analytics)
    params = sig.parameters
    kwargs: dict = {}
    if "tier_model" in params:
        kwargs["tier_model"] = tier_model
    if "classification_model" in params:
        kwargs["classification_model"] = classification_model
    if "xt_surface_mode" in params:
        kwargs["xt_surface_mode"] = xt_surface_mode
    if kwargs:
        return build_analytics(cache_version, **kwargs)
    if "impact_model" in params:
        return build_analytics(cache_version, impact_model=tier_model)
    return build_analytics(cache_version)


def _call_load_passes_grouped(
    cache_version: int,
    tier_model: str,
    classification_model: str,
    xt_surface_mode: str,
):
    sig = inspect.signature(load_passes_grouped)
    params = sig.parameters
    kwargs: dict = {}
    if "tier_model" in params:
        kwargs["tier_model"] = tier_model
    if "classification_model" in params:
        kwargs["classification_model"] = classification_model
    if "xt_surface_mode" in params:
        kwargs["xt_surface_mode"] = xt_surface_mode
    if kwargs:
        return load_passes_grouped(cache_version, **kwargs)
    if "impact_model" in params:
        return load_passes_grouped(cache_version, impact_model=tier_model)
    return load_passes_grouped(cache_version)


@st.cache_data(show_spinner=False)
def load_analytics(
    _cache_version: int = DATA_CACHE_VERSION,
    tier_model: str = TIER_MODEL_DEFAULT,
    classification_model: str = CLASSIFICATION_MODEL_DEFAULT,
    xt_surface_mode: str = FIXED_XT_SURFACE_MODE,
):
    return _call_build_analytics(
        _cache_version,
        normalize_tier_model(tier_model),
        normalize_classification_model(classification_model),
        normalize_xt_surface_mode(xt_surface_mode),
    )


@st.cache_data(show_spinner=False)
def load_passes(
    _cache_version: int = DATA_CACHE_VERSION,
    tier_model: str = TIER_MODEL_DEFAULT,
    classification_model: str = CLASSIFICATION_MODEL_DEFAULT,
    xt_surface_mode: str = FIXED_XT_SURFACE_MODE,
):
    return _call_load_passes_grouped(
        _cache_version,
        normalize_tier_model(tier_model),
        normalize_classification_model(classification_model),
        normalize_xt_surface_mode(xt_surface_mode),
    )


@st.cache_data(show_spinner=False)
def load_serie_a_passes(_cache_version: int = DATA_CACHE_VERSION):
    if not hasattr(pe, "load_serie_a_passes_grouped"):
        return {}
    return pe.load_serie_a_passes_grouped(
        _cache_version,
        tier_model=FIXED_TIER_MODEL,
        classification_model=FIXED_CLASSIFICATION_MODEL,
        xt_surface_mode=FIXED_XT_SURFACE_MODE,
    )


@st.cache_data(show_spinner=False)
def load_serie_a_players(_cache_version: int = DATA_CACHE_VERSION):
    if not hasattr(pe, "build_serie_a_players"):
        return []
    return pe.build_serie_a_players(
        _cache_version,
        tier_model=FIXED_TIER_MODEL,
        classification_model=FIXED_CLASSIFICATION_MODEL,
        xt_surface_mode=FIXED_XT_SURFACE_MODE,
    )


def _norm(s: str) -> str:
    return unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()


def rank_color(rank: int, total: int) -> str:
    """Score-based gradient: 9 green → 6 yellow → 3 red."""
    if total <= 0:
        return score_display_color(6.0)
    effective_rank = min(max(rank, 1), total)
    return score_display_color(rank_to_display_score(effective_rank, total))


def rating_value_color(pass_rating: float | None) -> str:
    if pass_rating is None:
        return "#334155"
    return score_display_color(float(pass_rating) * 10.0)


def _player_options(rated: list[dict]) -> list[tuple[str, str, str, str]]:
    rows = sorted(
        {(p["player_id"], p["player_name"], p.get("team", "—")) for p in rated},
        key=lambda x: _norm(x[1]),
    )
    return [(pid, name, team, f"{name} ({team})") for pid, name, team in rows]


def _sync_player_selection(
    players_by_id: dict[str, dict],
    label_by_id: dict[str, str],
) -> None:
    qp = st.query_params.get("player_id")
    if qp and qp in players_by_id:
        st.session_state["map_player_id"] = qp
        st.session_state[SELECTBOX_KEY] = label_by_id[qp]


def render_rating_table(
    rows: list[dict],
    *,
    selected_player_id: str | None,
) -> None:
    if not rows:
        st.info("Nenhum jogador elegível nesta posição.")
        return

    body = []
    for row in rows:
        pid = html.escape(str(row["player_id"]))
        rating = float(row["Rating"])
        rating_txt = fmt_rating_score(rating)
        sel = " sel" if selected_player_id and str(row["player_id"]) == str(selected_player_id) else ""
        body.append(
            f'<tr class="row{sel}" data-pid="{pid}" onclick="pickPlayer(\'{pid}\')">'
            f"<td>{html.escape(str(row['Jogador']))}</td>"
            f"<td class='team'>{html.escape(str(row['Time']))}</td>"
            f'<td class="rating">{rating_txt}</td>'
            "</tr>"
        )

    page = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><style>
*{{box-sizing:border-box}}
body{{margin:0;padding:0;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
  color:#e8edf5;background:transparent}}
.rx{{width:100%;border-collapse:separate;border-spacing:0;font-size:0.9rem;
  border:1px solid #2a3550;border-radius:10px;overflow:hidden}}
.rx th,.rx td{{padding:9px 12px;text-align:left;vertical-align:middle}}
.rx th{{background:linear-gradient(180deg,#1b2438,#141b2d);color:#8fa3bf;font-weight:600;
  font-size:0.72rem;letter-spacing:0.05em;text-transform:uppercase;border-bottom:1px solid #2f3b56}}
.rx td{{border-bottom:1px solid #232d42}}
.rx tr.row{{cursor:pointer;transition:background .15s ease}}
.rx tr.row:hover td{{background:#1a2238}}
.rx tr.row.sel td{{background:#1c3354}}
.rx tr.row.sel td:first-child{{box-shadow:inset 3px 0 0 #60a5fa}}
.rx tr:last-child td{{border-bottom:none}}
.team{{color:#9fb0c7}}
.rating{{font-weight:700;color:#dbeafe}}
</style>
<script>
function pickPlayer(pid) {{
  try {{
    const base = window.parent !== window ? window.parent : window;
    const url = new URL(base.location.href);
    url.searchParams.set("player_id", pid);
    base.location.href = url.toString();
  }} catch (e) {{
    const url = new URL(window.location.href);
    url.searchParams.set("player_id", pid);
    window.location.href = url.toString();
  }}
}}
</script></head><body>
<table class="rx"><thead><tr>
{"".join(f"<th>{html.escape(c)}</th>" for c in RATING_COLUMNS)}
</tr></thead><tbody>{"".join(body)}</tbody></table>
</body></html>"""

    height = min(44 * len(rows) + 52, 920)
    components.html(page, height=height, scrolling=False)


def _rating_warnings_html(player: dict) -> str:
    warnings: list[str] = []
    if not player.get("eligible_minutes", True):
        warnings.append("Menos de 30% dos minutos")
    if not player.get("eligible_passes", True):
        min_passes = player.get("position_min_passes")
        if min_passes is not None:
            min_txt = fmt_stat_value("passes_completed", min_passes)
            warnings.append(f"Menos de 30% dos passes da posição (mín. {min_txt})")
        else:
            warnings.append("Menos de 30% dos passes da posição")
    return "".join(
        '<span class="rating-warning-tip">'
        '<span class="rating-warning">⚠</span>'
        f'<span class="rating-tipbox">{html.escape(msg)}</span>'
        "</span>"
        for msg in warnings
    )


def _stat_display(player: dict, key: str) -> str:
    if key == "minutes_pct":
        pct = player.get("minutes_pct")
        return fmt_pct(pct * 100.0) if pct is not None else "—"
    return fmt_stat_value(key, player.get(key))


def _badge_text_color(hex_color: str) -> str:
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    return "#1e293b" if lum > 168 else "#f8fafc"


def _metric_label_html(key: str) -> str:
    label = analyst_metric_label(key)
    tip = html.escape(metric_tooltip(key))
    return (
        f'<span class="metric-tip">{html.escape(label)}'
        f'<span class="metric-tipbox">{tip}</span></span>'
    )


def _metric_rank_subtitle_html(player: dict, key: str, metric_ranks: dict) -> str:
    info = metric_ranks.get(key)
    if not info:
        return ""
    return (
        f'<span class="metric-rank-sub">'
        f'{html.escape(rank_in_group_label(int(info["rank"]), player.get("position_group")))}'
        f"</span>"
    )


def _metric_line_html(
    label: str,
    key: str,
    value: str,
    metric_ranks: dict,
    *,
    player: dict | None = None,
    show_rank: bool = True,
) -> str:
    badge = ""
    if show_rank:
        info = metric_ranks.get(key)
        if info:
            rank = int(info["rank"])
            total = int(info["total"])
            color = rank_color(rank, total)
            badge = (
                f'<span class="rank-tip">'
                f'<span class="rank-badge" style="background:{color}"></span>'
                f'<span class="rank-tipbox">{rank}/{total}</span>'
                f"</span>"
            )
    rank_sub = _metric_rank_subtitle_html(player or {}, key, metric_ranks) if show_rank and player else ""
    value_inner = (
        f'<span class="val-wrap">{badge}<span class="stat-val">{html.escape(value)}</span></span>'
        f"{rank_sub}"
        if badge
        else f'<span class="stat-val">{html.escape(value)}</span>{rank_sub}'
    )
    label_html = _metric_label_html(key) if key else html.escape(label)
    return (
        '<div class="metric-line">'
        f"<span>{label_html}</span>"
        f'<span style="text-align:right">{value_inner}</span>'
        "</div>"
    )


def _section_header_html(title: str, section_key: str, player: dict) -> str:
    section_ratings = player.get("section_ratings") if isinstance(player.get("section_ratings"), dict) else {}
    section_rank_info = player.get("section_rating_ranks") if isinstance(player.get("section_rating_ranks"), dict) else {}
    score = section_ratings.get(section_key)
    pill = ""
    if score is not None:
        txt = fmt_rating_score(score)
        rank_info = section_rank_info.get(section_key)
        if rank_info:
            color = rank_color(int(rank_info["rank"]), int(rank_info["total"]))
            txt_color = _badge_text_color(color)
            rank_txt = f'{int(rank_info["rank"])}/{int(rank_info["total"])}'
            pill = (
                f'<span class="section-rating-tip">'
                f'<span class="section-rating-pill" style="background:{color};color:{txt_color}">'
                f"{html.escape(txt)}</span>"
                f'<span class="rating-tipbox">{rank_txt}</span>'
                f"</span>"
            )
        else:
            pill = f'<span class="section-rating-pill">{html.escape(txt)}</span>'
    return (
        '<div class="stat-section-row">'
        f'<span class="stat-section">{html.escape(title)}</span>'
        f"{pill}"
        "</div>"
    )


def _build_sections_html(
    player: dict,
    metric_ranks: dict,
    sections: list[tuple[str, str | None, tuple[str, ...], bool]],
) -> str:
    parts: list[str] = []
    for title, section_key, keys, show_rank in sections:
        if section_key:
            parts.append(_section_header_html(title, section_key, player))
        else:
            parts.append(
                f'<div class="stat-section-row"><span class="stat-section">{html.escape(title)}</span></div>'
            )
        for key in keys:
            parts.append(
                _metric_line_html(
                    analyst_metric_label(key),
                    key,
                    _stat_display(player, key),
                    metric_ranks,
                    player=player,
                    show_rank=show_rank,
                )
            )
    return "".join(parts)


def _rating_header_html(player: dict, metric_ranks: dict) -> str:
    rating_val = player.get("pass_rating")
    rating_txt = fmt_rating_score(rating_val) if rating_val is not None else "—"
    rating_info = metric_ranks.get("pass_rating")
    is_solo = bool(player.get("rating_is_solo"))

    if rating_info and rating_val is not None:
        r_color = rating_value_color(rating_val)
        r_txt = _badge_text_color(r_color)
        rank_txt = f'{int(rating_info["rank"])}/{int(rating_info["total"])}'
        if is_solo:
            rank_txt += " · individual"
        elif player.get("rating_is_compared"):
            rank_txt += " · vs aptos"
        rating_box = (
            f'<span class="rating-tip">'
            f'<div class="rating-box" style="background:{r_color};color:{r_txt};margin-bottom:0">'
            f"{html.escape(rating_txt)}</div>"
            f'<span class="rating-tipbox">{html.escape(rank_txt)}</span>'
            f"</span>"
        )
    else:
        rating_box = (
            f'<div class="rating-box" style="background:#334155;color:#f8fafc;margin-bottom:0">'
            f"{html.escape(rating_txt)}</div>"
        )

    warnings = _rating_warnings_html(player)
    return f'<div class="rating-row">{rating_box}{warnings}</div>'


def _section_grade_card_html(
    player: dict,
    section_key: str,
    title: str,
    subtitle: str,
) -> str:
    section_ratings = player.get("section_ratings") if isinstance(player.get("section_ratings"), dict) else {}
    section_rank_info = player.get("section_rating_ranks") if isinstance(player.get("section_rating_ranks"), dict) else {}
    score = section_ratings.get(section_key)
    score_html = '<span class="section-rating-pill" style="background:#334155;color:#f8fafc">—</span>'
    rank_html = ""
    if score is not None:
        txt = fmt_rating_score(score)
        rank_info = section_rank_info.get(section_key)
        if rank_info:
            color = rank_color(int(rank_info["rank"]), int(rank_info["total"]))
            txt_color = _badge_text_color(color)
            score_html = (
                f'<span class="section-rating-pill" style="background:{color};color:{txt_color}">'
                f"{html.escape(txt)}</span>"
            )
            rank_html = (
                f'<div class="grade-card-rank">'
                f'{html.escape(rank_in_group_label(int(rank_info["rank"]), player.get("position_group")))}'
                f"</div>"
            )
        else:
            score_html = f'<span class="section-rating-pill">{html.escape(txt)}</span>'
    return (
        '<div class="grade-card">'
        f'<div class="grade-card-title">{html.escape(title)}</div>'
        f'<div class="grade-card-sub">{html.escape(subtitle)}</div>'
        f'<div class="grade-card-score">{score_html}</div>'
        f"{rank_html}"
        "</div>"
    )


def _section_metrics_card_html(
    player: dict,
    section_key: str,
    title: str,
    keys: tuple[str, ...],
) -> str:
    metric_ranks = player.get("metric_ranks") if isinstance(player.get("metric_ranks"), dict) else {}
    lines = "".join(
        _metric_line_html(
            analyst_metric_label(key),
            key,
            _stat_display(player, key),
            metric_ranks,
            player=player,
            show_rank=True,
        )
        for key in keys
    )
    return (
        '<div class="player-card">'
        f'<div class="stat-section-row"><span class="stat-section">{html.escape(title)}</span></div>'
        f"{lines}"
        "</div>"
    )


def _metric_section_state_key(player_id: str) -> str:
    return f"open_metric_section_{player_id}"


def _cmp_delta_html(target_val: float | None, similar_val: float | None) -> tuple[str, str]:
    if target_val is None or similar_val is None:
        return "", ""
    t = float(target_val)
    s = float(similar_val)
    if abs(t - s) < 0.05:
        dot = '<span class="cmp-delta flat" title="Empate">●</span>'
        return dot, dot
    if t > s:
        return (
            '<span class="cmp-delta up" title="Acima do similar">▲</span>',
            '<span class="cmp-delta down" title="Abaixo da referência">▼</span>',
        )
    return (
        '<span class="cmp-delta down" title="Abaixo do similar">▼</span>',
        '<span class="cmp-delta up" title="Acima da referência">▲</span>',
    )


def render_player_layout(player: dict, passes) -> None:
    team_label = player.get("team", "—")
    player_id = str(player.get("player_id", ""))
    col_map1, col_map2, col_map3 = st.columns(3, gap="small")

    if passes is None or passes.empty:
        with col_map1:
            st.warning("Sem passes para este jogador.")
    else:
        with col_map1:
            st.caption("Passes completos — todos no campo")
            fig_all = draw_all_completed_passes_map(
                passes,
                player["player_name"],
                team_label,
                compact=False,
            )
            st.pyplot(fig_all, clear_figure=True, use_container_width=True)
        with col_map2:
            st.caption("Passes de impacto")
            fig = draw_impact_pass_map(passes, player["player_name"], team_label, compact=False)
            st.pyplot(fig, clear_figure=True, use_container_width=True)
        with col_map3:
            st.caption("Destino — heatmap")
            fig_heat = draw_pass_destination_heatmap(passes, player["player_name"], team_label, compact=False)
            st.pyplot(fig_heat, clear_figure=True, use_container_width=True)

    general_sections: list[tuple[str, str | None, tuple[str, ...], bool]] = [
        (
            "Participação",
            None,
            (
                "minutes",
                "passes_completed",
                "minutes_pct",
                "impact_passes",
                "high_impact_passes",
            ),
            False,
        ),
    ]

    metric_ranks = player.get("metric_ranks") if isinstance(player.get("metric_ranks"), dict) else {}
    general_card = (
        '<div class="player-card player-info-card">'
        f"<h3>{html.escape(player['player_name'])}</h3>"
        f'<div class="sub">{html.escape(player.get("team", "—"))} · {html.escape(str(player.get("position", "—")))}</div>'
        f"{_rating_header_html(player, metric_ranks)}"
        + _build_sections_html(player, metric_ranks, general_sections)
        + "</div>"
    )
    st.markdown(general_card, unsafe_allow_html=True)

    st.markdown("##### Pilares de avaliação")
    grade_cols = st.columns(len(SCOUT_SECTION_SPECS), gap="small")
    open_section = st.session_state.get(_metric_section_state_key(player_id))
    for col, (section_key, title, subtitle, _keys) in zip(grade_cols, SCOUT_SECTION_SPECS):
        with col:
            st.markdown(
                _section_grade_card_html(player, section_key, title, subtitle),
                unsafe_allow_html=True,
            )
            if st.button(
                "Ver métricas" if open_section != section_key else "Selecionado",
                key=f"grade_btn_{player_id}_{section_key}",
                use_container_width=True,
            ):
                st.session_state[_metric_section_state_key(player_id)] = section_key

    if open_section:
        spec_by_key = {item[0]: item for item in SCOUT_SECTION_SPECS}
        if open_section in spec_by_key:
            _section_key, title, _subtitle, keys = spec_by_key[open_section]
            c_close, _ = st.columns([1, 5])
            with c_close:
                if st.button("Fechar", key=f"grade_close_{player_id}"):
                    st.session_state.pop(_metric_section_state_key(player_id), None)
                    st.rerun()
            st.markdown(
                _section_metrics_card_html(player, _section_key, title, keys),
                unsafe_allow_html=True,
            )


def render_map_section(
    all_players: list[dict],
    players_by_id: dict[str, dict],
    pool_by_position: dict[str, list[dict]],
    passes_by_player: dict,
) -> None:
    st.subheader("Mapa — passes de impacto")
    st.caption("Clique em um jogador na tabela de rating ou selecione abaixo.")

    options = _player_options(all_players)
    if not options:
        st.info("Nenhum jogador com passes para o mapa.")
        return

    labels = [o[3] for o in options]
    id_by_label = {o[3]: o[0] for o in options}
    label_by_id = {o[0]: o[3] for o in options}

    _sync_player_selection(players_by_id, label_by_id)

    selected_label = st.selectbox(
        "Jogador",
        options=labels,
        key=SELECTBOX_KEY,
        placeholder="Selecione um jogador",
    )

    if not selected_label:
        st.info("Selecione um jogador na lista ou clique em uma linha da tabela de rating.")
        return

    player_id = id_by_label[selected_label]
    st.session_state["map_player_id"] = player_id
    player = dict(players_by_id[player_id])
    if not player.get("eligible_for_rating"):
        group = str(player.get("position_group") or "—")
        player = rate_player_vs_eligible_pool(player, pool_by_position.get(group, []))
    passes = passes_by_player.get(player_id)

    render_player_layout(player, passes)


def render_rating_section(rated: list[dict], *, selected_player_id: str | None) -> None:
    st.subheader("Rating por grupo de posição")
    st.caption(
        "Rating = média das notas por métrica no grupo (1º = 9,0 · mediano = 6,0 · último = 3,0). "
        f"Elegível: >{int(RATING_MIN_MINUTES_PCT * 100)}% dos minutos e ≥{int(RATING_MIN_PASSES_PCT * 100)}% dos passes do grupo. "
        "Fora do pool: rating comparado aos aptos ao selecionar o jogador."
    )
    for group in POSITION_GROUPS_ORDER:
        subset = sorted(
            [p for p in rated if p["position_group"] == group],
            key=lambda p: p.get("pass_rating", 0),
            reverse=True,
        )[:RATING_TOP_N]
        if not subset:
            continue
        with st.expander(f"{group} ({len(subset)})", expanded=group == "Zagueiros"):
            rows = [
                {
                    "player_id": p["player_id"],
                    "Jogador": p["player_name"],
                    "Time": p["team"],
                    "Rating": p["pass_rating"],
                    "metric_ranks": p.get("metric_ranks", {}),
                }
                for p in subset
            ]
            render_rating_table(
                rows,
                selected_player_id=selected_player_id,
            )


def _group_players_by_position_group(players: list[dict]) -> dict[str, list[dict]]:
    by_group: dict[str, list[dict]] = {}
    for player in players:
        group = str(player.get("position_group") or "—")
        by_group.setdefault(group, []).append(player)
    return by_group


def _eligible_pool_for_player(player: dict, pool_by_group: dict[str, list[dict]]) -> list[dict]:
    group = str(player.get("position_group") or "—")
    pool = pool_by_group.get(group, [])
    eligible = [p for p in pool if p.get("eligible_for_rating")]
    return eligible if eligible else list(pool)


def _player_metric_ranks(player: dict, pool_by_group: dict[str, list[dict]]) -> dict:
    pool = _eligible_pool_for_player(player, pool_by_group)
    if not pool:
        return {}
    if player.get("metric_ranks") and player.get("player_id") in {p["player_id"] for p in pool}:
        return dict(player.get("metric_ranks") or {})
    rated = rate_player_vs_eligible_pool(player, pool)
    return dict(rated.get("metric_ranks") or {})


def _comparison_metrics_html(
    target: dict,
    similar: dict,
    *,
    target_league: str,
    similar_league: str,
    target_pct: dict[str, float],
    similar_pct: dict[str, float],
    target_ranks: dict,
    similar_ranks: dict,
) -> str:
    rows = [
        '<div class="player-card">',
        '<div class="cmp-row cmp-row-head">',
        "<span>Métrica</span>",
        f"<span>{html.escape(target_league)}</span>",
        f"<span>{html.escape(similar_league)}</span>",
        "</div>",
    ]
    for section_name, section_keys in sim.SIMILARITY_COMPARE_SECTIONS:
        rows.append(f'<div class="cmp-section-title">{html.escape(section_name)}</div>')
        for key in section_keys:
            label = _metric_label_html(key)
            t_rank = _metric_rank_subtitle_html(target, key, target_ranks)
            s_rank = _metric_rank_subtitle_html(similar, key, similar_ranks)
            t_delta, s_delta = _cmp_delta_html(target_pct.get(key), similar_pct.get(key))
            t_val = html.escape(sim.fmt_percentile_value(target_pct.get(key)))
            s_val = html.escape(sim.fmt_percentile_value(similar_pct.get(key)))
            rows.extend([
                '<div class="cmp-row">',
                f'<span class="cmp-cell-label">{label}</span>',
                (
                    f'<span><span class="cmp-value-wrap">'
                    f'<span class="cmp-cell-value">{t_val}</span>{t_delta}</span>{t_rank}</span>'
                ),
                (
                    f'<span><span class="cmp-value-wrap">'
                    f'<span class="cmp-cell-value">{s_val}</span>{s_delta}</span>{s_rank}</span>'
                ),
                "</div>",
            ])
    rows.append("</div>")
    return "".join(rows)


def _render_comparison_maps_row(
    target: dict,
    similar: dict,
    target_passes,
    similar_passes,
    *,
    target_league: str,
    similar_league: str,
) -> None:
    m1, m2 = st.columns(2, gap="small")
    name_t = str(target.get("player_name", "—"))
    name_s = str(similar.get("player_name", "—"))
    grid_label = f"{sim.ORIGIN_ANALYSIS_COLS}×{sim.ORIGIN_ANALYSIS_ROWS}"
    with m1:
        st.caption(f"Origem · {name_t} · {grid_label}")
        if target_passes is not None and not target_passes.empty:
            fig = draw_pass_origin_heatmap(
                target_passes,
                name_t,
                str(target.get("team", "—")),
                cols=sim.ORIGIN_ANALYSIS_COLS,
                rows=sim.ORIGIN_ANALYSIS_ROWS,
                compare=True,
            )
            st.pyplot(fig, clear_figure=True, use_container_width=True)
        else:
            st.caption("Sem passes.")
    with m2:
        st.caption(f"Origem · {name_s} · {grid_label}")
        if similar_passes is not None and not similar_passes.empty:
            fig = draw_pass_origin_heatmap(
                similar_passes,
                name_s,
                str(similar.get("team", "—")),
                cols=sim.ORIGIN_ANALYSIS_COLS,
                rows=sim.ORIGIN_ANALYSIS_ROWS,
                compare=True,
            )
            st.pyplot(fig, clear_figure=True, use_container_width=True)
        else:
            st.caption("Sem passes.")


def render_presentation_tab(
    all_players: list[dict],
    passes_by_player: dict,
) -> None:
    st.subheader("Apresentação")
    st.markdown(
        "Guia rápido do **Passes xTh**: o que cada visual mostra e como interpretar "
        "os números na prática de scouting."
    )

    st.markdown(
        '<div class="pres-card"><h4>O que é este dashboard?</h4>'
        "<p>Medimos a qualidade dos passes com um modelo de <strong>expected threat (xT)</strong>. "
        "Passes que aumentam a probabilidade de gol valem mais. O rating resume o jogador "
        "frente aos pares da <strong>mesma posição</strong> na Série B.</p></div>",
        unsafe_allow_html=True,
    )

    example = next(
        (
            p for p in all_players
            if passes_by_player.get(str(p["player_id"])) is not None
            and not passes_by_player[str(p["player_id"])].empty
        ),
        None,
    )
    if example:
        ex_id = str(example["player_id"])
        ex_passes = passes_by_player[ex_id]
        ex_name = str(example.get("player_name", "Jogador"))
        st.markdown("#### Exemplo visual — três mapas")
        st.caption(f"Referência: {ex_name} ({example.get('team', '—')})")
        c1, c2, c3 = st.columns(3, gap="small")
        with c1:
            st.markdown(
                '<div class="pres-card"><h4>1 · Passes completos</h4>'
                "<p>Cada passe <em>completado</em> no campo — origem e trajeto. "
                "Mostra onde o jogador circula com a bola nos pés.</p></div>",
                unsafe_allow_html=True,
            )
            fig = draw_all_completed_passes_map(
                ex_passes, ex_name, str(example.get("team", "—")), compact=False,
            )
            st.pyplot(fig, clear_figure=True, use_container_width=True)
        with c2:
            st.markdown(
                '<div class="pres-card"><h4>2 · Passes de impacto</h4>'
                "<p>Subset que muda o xT de forma relevante. "
                "Cores destacam progressão e alto impacto.</p></div>",
                unsafe_allow_html=True,
            )
            fig = draw_impact_pass_map(ex_passes, ex_name, str(example.get("team", "—")), compact=False)
            st.pyplot(fig, clear_figure=True, use_container_width=True)
        with c3:
            st.markdown(
                '<div class="pres-card"><h4>3 · Destino (heatmap)</h4>'
                "<p>Para onde os passes de impacto <em>chegam</em>. "
                "Útil para ver penetração e zonas de recepção.</p></div>",
                unsafe_allow_html=True,
            )
            fig = draw_pass_destination_heatmap(
                ex_passes, ex_name, str(example.get("team", "—")), compact=False,
            )
            st.pyplot(fig, clear_figure=True, use_container_width=True)

    st.markdown("#### Pilares de avaliação")
    pillar_lines = "".join(
        f"<li><strong>{html.escape(title)}</strong> — {html.escape(subtitle)}</li>"
        for _key, title, subtitle, _keys in SCOUT_SECTION_SPECS
    )
    st.markdown(
        '<div class="pres-card"><h4>Cards com nota por pilar</h4>'
        f"<p>Cada pilar tem nota 0–10 e rank no grupo. Clique em <em>Ver métricas</em> "
        f"para abrir o detalhe:</p><ul style='margin:0.5rem 0 0 1rem;color:#94a3b8;"
        f"font-size:0.88rem;line-height:1.5'>{pillar_lines}</ul></div>",
        unsafe_allow_html=True,
    )

    st.markdown("#### Métricas e rating")
    col_m1, col_m2 = st.columns(2, gap="medium")
    with col_m1:
        st.markdown(
            '<div class="pres-card"><h4>Leitura das métricas</h4>'
            "<p>Nomes em linguagem de scout; passe o mouse para ver a definição. "
            "Abaixo de cada valor: <em>23º em Laterais</em> no grupo de posição.</p></div>",
            unsafe_allow_html=True,
        )
    with col_m2:
        st.markdown(
            '<div class="pres-card"><h4>Rating geral (0–10)</h4>'
            "<p>Média dos ranks nas métricas principais. "
            "Verde = topo do grupo; amarelo = meio; vermelho = abaixo.</p></div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="pres-card"><h4>Similaridade B ↔ A</h4>'
            "<p>Compare jogadores entre ligas na mesma posição detalhada. "
            "Na comparação: dois mapas 12×8 de origem e tabela com ▲/▼ "
            "verde/vermelho entre os percentis.</p></div>",
            unsafe_allow_html=True,
        )

    st.markdown("#### Como usar")
    steps = [
        ("Apresentação", "Entenda mapas, pilares e o fluxo de leitura."),
        ("Dashboard", "Escolha o jogador; abra os pilares que quiser detalhar."),
        ("Similaridade", "Selecione um atleta e compare com similares da outra liga."),
    ]
    for idx, (title, text) in enumerate(steps, start=1):
        st.markdown(
            f'<div class="pres-step"><span class="pres-step-num">{idx}</span>'
            f"<div><strong>{html.escape(title)}</strong><br>"
            f'<span style="color:#94a3b8;font-size:0.88rem">{html.escape(text)}</span></div></div>',
            unsafe_allow_html=True,
        )


def _render_similarity_player_panel(
    player: dict,
    passes,
    *,
    league: str,
    similarity_pct: float | None = None,
    comparison_mode: bool = False,
) -> None:
    header = (
        f"**{html.escape(str(player.get('player_name', '—')))}** · "
        f"{html.escape(str(player.get('team', '—')))} · "
        f"{html.escape(str(player.get('position', '—')))}"
    )
    if similarity_pct is not None:
        header += f" · sim. **{similarity_pct:.1f}%**"
    st.markdown(header, unsafe_allow_html=True)

    if not comparison_mode:
        m1, m2, m3 = st.columns(3)
        m1.metric("Minutos", fmt_stat_value("minutes", player.get("minutes")))
        m2.metric("Passes", fmt_stat_value("passes_completed", player.get("passes_completed")))
        m3.metric("Impact p90", fmt_stat_value("impact_passes_p90", player.get("impact_passes_p90")))
    else:
        g1, g2 = st.columns(2)
        g1.metric("Minutos", fmt_stat_value("minutes", player.get("minutes")))
        g2.metric("Passes", fmt_stat_value("passes_completed", player.get("passes_completed")))

    profile = sim.pass_origin_profile(passes) if passes is not None else None
    if profile is not None and not comparison_mode:
        st.caption(f"Origem dominante: {sim.describe_dominant_origin_zone(profile)}")

    if not comparison_mode and passes is not None and not passes.empty:
        fig = draw_pass_origin_heatmap(
            passes,
            str(player.get("player_name", "—")),
            str(player.get("team", "—")),
            cols=sim.ORIGIN_GRID_COLS,
            rows=sim.ORIGIN_GRID_ROWS,
            compare=comparison_mode,
            tiny=not comparison_mode,
        )
        st.pyplot(fig, clear_figure=True, use_container_width=comparison_mode)
    else:
        st.caption("Sem passes para heatmap de origem.")


def _similarity_results_df(
    results: list[dict],
    *,
    include_origin: bool = False,
    origin_dual: bool = False,
    origin_column: bool = False,
):
    import pandas as pd

    origin_col_label = (
        f"Sim. origem ({sim.ORIGIN_ANALYSIS_COLS}×{sim.ORIGIN_ANALYSIS_ROWS})"
    )
    rows = []
    for rank, row in enumerate(results, start=1):
        entry = {
            "#": rank,
            "Jogador": row.get("player_name", "—"),
            "Time": row.get("team", "—"),
            "Posição": row.get("position", "—"),
            "_player_id": str(row.get("player_id", "")),
        }
        if origin_dual:
            entry["Sim. métricas"] = f"{row.get('similarity_pct', 0):.1f}%"
            entry["Sim. origem"] = f"{row.get('origin_similarity_pct', 0):.1f}%"
            entry["Origem dominante"] = row.get("origin_dominant", "—")
        elif include_origin:
            entry["Similaridade"] = f"{row.get('similarity_pct', 0):.1f}%"
            entry["Origem dominante"] = row.get("origin_dominant", "—")
        else:
            entry["Similaridade"] = f"{row.get('similarity_pct', 0):.1f}%"
            if origin_column:
                origin_val = row.get("origin_similarity_pct")
                entry[origin_col_label] = (
                    f"{float(origin_val):.1f}%" if origin_val is not None else "—"
                )
            entry["Impact p90"] = fmt_stat_value("impact_passes_p90", row.get("impact_passes_p90"))
            entry["PHI p90"] = fmt_stat_value("phi_p90", row.get("phi_p90"))
            entry["ΔxT p90"] = fmt_stat_value("dxt_p90", row.get("dxt_p90"))
        rows.append(entry)
    return pd.DataFrame(rows)


def _render_similarity_results_tab(
    *,
    results: list[dict],
    target: dict,
    target_passes,
    pool_passes: dict,
    target_league: str,
    similar_league: str,
    target_pool_by_pos: dict[str, list[dict]],
    similar_pool_by_pos: dict[str, list[dict]],
    target_pool_by_group: dict[str, list[dict]],
    similar_pool_by_group: dict[str, list[dict]],
    pick_key: str,
    include_origin: bool = False,
    origin_dual: bool = False,
    origin_column: bool = False,
) -> None:
    import pandas as pd

    if not results:
        st.info("Nenhum similar encontrado.")
        return

    df = _similarity_results_df(
        results,
        include_origin=include_origin,
        origin_dual=origin_dual,
        origin_column=origin_column,
    )
    display_df = df.drop(columns=["_player_id"])
    pick = st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key=pick_key,
    )

    selected_rows: list[int] = []
    if pick is not None:
        selection = getattr(pick, "selection", None)
        if selection is not None:
            selected_rows = list(getattr(selection, "rows", []) or [])
        elif isinstance(pick, dict):
            selected_rows = list(pick.get("selection", {}).get("rows", []) or [])
    if not selected_rows and pick_key in st.session_state:
        state = st.session_state.get(pick_key)
        if isinstance(state, dict):
            selected_rows = list(state.get("selection", {}).get("rows", []) or [])

    if not selected_rows:
        st.caption("Clique em uma linha da tabela para comparar com o jogador selecionado.")
        return

    similar = dict(results[int(selected_rows[0])])
    similar_id = str(similar.get("player_id", ""))
    similar_passes = pool_passes.get(similar_id)

    compare_keys = sim.SIMILARITY_METRICS_A
    target_pct = sim.position_pool_percentiles(target, target_pool_by_pos, keys=compare_keys)
    similar_pct = sim.position_pool_percentiles(similar, similar_pool_by_pos, keys=compare_keys)
    target_ranks = _player_metric_ranks(target, target_pool_by_group)
    similar_ranks = _player_metric_ranks(similar, similar_pool_by_group)
    target_pos = sim.player_search_position(target) or "—"
    similar_pos = sim.player_search_position(similar) or "—"

    st.markdown("#### Comparação")
    st.caption(
        f"Percentis na posição detalhada · ranks no grupo · ▲ verde = acima · ▼ vermelho = abaixo "
        f"({html.escape(target_pos)} · {html.escape(target_league)} vs "
        f"{html.escape(similar_pos)} · {html.escape(similar_league)})."
    )

    _render_comparison_maps_row(
        target,
        similar,
        target_passes,
        similar_passes,
        target_league=target_league,
        similar_league=similar_league,
    )

    col_target, col_similar = st.columns(2, gap="small")
    with col_target:
        st.markdown(f"**Referência · {html.escape(target_league)}**", unsafe_allow_html=True)
        _render_similarity_player_panel(
            target,
            target_passes,
            league=target_league,
            comparison_mode=True,
        )
    with col_similar:
        st.markdown(f"**Similar · {html.escape(similar_league)}**", unsafe_allow_html=True)
        _render_similarity_player_panel(
            similar,
            similar_passes,
            league=similar_league,
            similarity_pct=float(similar.get("similarity_pct") or 0),
            comparison_mode=True,
        )
        if similar.get("origin_similarity_pct") is not None:
            st.caption(
                f"Similaridade de origem ({sim.ORIGIN_ANALYSIS_COLS}×{sim.ORIGIN_ANALYSIS_ROWS}): "
                f"{float(similar['origin_similarity_pct']):.1f}%"
            )

    st.markdown(
        _comparison_metrics_html(
            target,
            similar,
            target_league=target_league,
            similar_league=similar_league,
            target_pct=target_pct,
            similar_pct=similar_pct,
            target_ranks=target_ranks,
            similar_ranks=similar_ranks,
        ),
        unsafe_allow_html=True,
    )


def render_similarity_section(
    all_players: list[dict],
    passes_by_player_sb: dict,
    serie_a_passes: dict,
    *,
    sb_to_sa: bool,
) -> None:
    import pandas as pd

    title = "Similaridade B → A" if sb_to_sa else "Similaridade A → B"
    st.subheader(title)
    st.caption(
        f"Selecione um jogador da {'Série B' if sb_to_sa else 'Série A'}; "
        f"a tabela mostra os top {SIMILARITY_TOP_K} da {'Série A' if sb_to_sa else 'Série B'} "
        "na mesma posição detalhada. Clique em uma linha para comparar."
    )

    if not all_players:
        st.info("Nenhum jogador disponível.")
        return

    serie_a_players = load_serie_a_players()
    if not serie_a_players:
        st.warning(
            "Dados da Série A indisponíveis — confirme season_all_brfull.csv e reimplante o app."
        )
        return

    sb_enriched = enrich_player_eligibility(all_players)
    serie_a_enriched = enrich_player_eligibility(serie_a_players)
    prefix = "ba" if sb_to_sa else "ab"
    serie_a_by_pos = sim.group_players_by_detailed_position(serie_a_enriched)
    sb_by_pos = sim.group_players_by_detailed_position(sb_enriched)
    serie_a_by_group = _group_players_by_position_group(serie_a_enriched)
    sb_by_group = _group_players_by_position_group(sb_enriched)
    players_sb_by_id = {str(p["player_id"]): p for p in sb_enriched}
    players_sa_by_id = {str(p["player_id"]): p for p in serie_a_enriched}

    if sb_to_sa:
        options = _player_options(sb_enriched)
        select_label = "Jogador Série B"
        select_key = SIMILARITY_SELECT_SB_KEY
    else:
        options = _player_options(serie_a_enriched)
        select_label = "Jogador Série A"
        select_key = SIMILARITY_SELECT_SA_KEY

    if not options:
        st.info("Nenhum jogador disponível para similaridade.")
        return

    labels = [o[3] for o in options]
    id_by_label = {o[3]: o[0] for o in options}
    selected_label = st.selectbox(
        select_label,
        options=labels,
        key=select_key,
        placeholder="Selecione um jogador",
    )
    if not selected_label:
        st.info("Selecione um jogador para ver os similares.")
        return

    target_id = id_by_label[selected_label]
    search_pos = sim.player_search_position(
        players_sb_by_id[target_id] if sb_to_sa else players_sa_by_id[target_id]
    )
    if sb_to_sa:
        target = dict(players_sb_by_id[target_id])
        target_passes = passes_by_player_sb.get(target_id)
        pool = sim.similarity_search_pool(serie_a_by_pos, search_pos)
        full_dest_pool = sim.outfield_players(serie_a_enriched)
        pool_passes = serie_a_passes
        pool_label = f"Série A · {search_pos or '—'}"
        origin_pool_label = f"Série A · origem similar (todas posições, top {sim.ORIGIN_PREFILTER_TOP_N})"
        target_league = "Série B"
    else:
        target = dict(players_sa_by_id[target_id])
        target_passes = serie_a_passes.get(target_id)
        pool = sim.similarity_search_pool(sb_by_pos, search_pos)
        full_dest_pool = sim.outfield_players(sb_enriched)
        pool_passes = passes_by_player_sb
        pool_label = f"Série B · {search_pos or '—'}"
        origin_pool_label = f"Série B · origem similar (todas posições, top {sim.ORIGIN_PREFILTER_TOP_N})"
        target_league = "Série A"

    if not search_pos:
        st.warning("Posição inválida para comparação (goleiros são excluídos).")
        return

    if not pool:
        st.warning(
            f"Nenhum jogador elegível na posição **{html.escape(search_pos)}** em {pool_label.split(' · ')[0]}."
        )
        return

    st.markdown(
        f"**{html.escape(str(target.get('player_name', '—')))}** · "
        f"{html.escape(str(target.get('team', '—')))} · "
        f"{html.escape(str(target.get('position', '—')))} · "
        f"{html.escape(target_league)} → pool **{html.escape(pool_label)}** ({len(pool)} jogadores)",
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns(2)
    c1.metric("Minutos", fmt_stat_value("minutes", target.get("minutes")))
    c2.metric("Passes", fmt_stat_value("passes_completed", target.get("passes_completed")))

    tab_a, tab_c, tab_origin = st.tabs(
        ["Opção A — percentil", "Opção C — z-score", "Origem dos passes"]
    )

    top_k = SIMILARITY_TOP_K
    target_league_label = target_league
    similar_league_label = "Série A" if sb_to_sa else "Série B"

    with tab_a:
        st.caption(
            f"Distância euclidiana no perfil percentil (0–100) dentro do pool {similar_league_label}."
        )
        results_a = sim.find_similar_option_a(target, pool, top_k=top_k)
        _render_similarity_results_tab(
            results=results_a,
            target=target,
            target_passes=target_passes,
            pool_passes=pool_passes,
            target_league=target_league_label,
            similar_league=similar_league_label,
            target_pool_by_pos=sb_by_pos if sb_to_sa else serie_a_by_pos,
            similar_pool_by_pos=serie_a_by_pos if sb_to_sa else sb_by_pos,
            target_pool_by_group=sb_by_group if sb_to_sa else serie_a_by_group,
            similar_pool_by_group=serie_a_by_group if sb_to_sa else sb_by_group,
            pick_key=f"sim_{prefix}_pick_a",
            include_origin=False,
        )
        with st.expander("Métricas usadas (Opção A)"):
            st.write(", ".join(metric_label(k) for k in sim.SIMILARITY_METRICS_A))

    with tab_c:
        st.caption(
            f"Distância euclidiana ponderada em z-scores do pool {similar_league_label}. "
            f"A coluna **Sim. origem ({sim.ORIGIN_ANALYSIS_COLS}×{sim.ORIGIN_ANALYSIS_ROWS})** "
            "é informativa (cosseno entre mapas de origem) e não altera o ranking."
        )
        results_c = sim.find_similar_option_c(target, pool, top_k=top_k)
        results_c = sim.attach_pass_origin_similarity(
            results_c,
            target_passes,
            pool_passes,
        )
        _render_similarity_results_tab(
            results=results_c,
            target=target,
            target_passes=target_passes,
            pool_passes=pool_passes,
            target_league=target_league_label,
            similar_league=similar_league_label,
            target_pool_by_pos=sb_by_pos if sb_to_sa else serie_a_by_pos,
            similar_pool_by_pos=serie_a_by_pos if sb_to_sa else sb_by_pos,
            target_pool_by_group=sb_by_group if sb_to_sa else serie_a_by_group,
            similar_pool_by_group=serie_a_by_group if sb_to_sa else sb_by_group,
            pick_key=f"sim_{prefix}_pick_c",
            include_origin=False,
            origin_column=True,
        )
        with st.expander("Métricas usadas (Opção C)"):
            st.write(", ".join(metric_label(k) for k in sim.SIMILARITY_METRICS_A))

    with tab_origin:
        st.caption(
            f"Dupla similaridade: (1) top {sim.ORIGIN_PREFILTER_TOP_N} jogadores de "
            f"{similar_league_label} com origem de passes mais parecida (qualquer posição); "
            f"(2) entre eles, ranking por perfil percentil das métricas (Opção A)."
        )
        if sim.pass_origin_profile(target_passes) is None:
            st.warning("Sem passes completos suficientes para perfil de origem do jogador selecionado.")
            return
        if not pool_passes or not full_dest_pool:
            st.warning("Passes do pool indisponíveis para comparação espacial.")
            return
        st.caption(f"Pool etapa 1: **{origin_pool_label}** ({len(full_dest_pool)} elegíveis)")
        results_origin = sim.find_similar_origin_then_percentile(
            target,
            target_passes,
            full_dest_pool,
            pool_passes,
            top_k=top_k,
        )
        _render_similarity_results_tab(
            results=results_origin,
            target=target,
            target_passes=target_passes,
            pool_passes=pool_passes,
            target_league=target_league_label,
            similar_league=similar_league_label,
            target_pool_by_pos=sb_by_pos if sb_to_sa else serie_a_by_pos,
            similar_pool_by_pos=serie_a_by_pos if sb_to_sa else sb_by_pos,
            target_pool_by_group=sb_by_group if sb_to_sa else serie_a_by_group,
            similar_pool_by_group=serie_a_by_group if sb_to_sa else sb_by_group,
            pick_key=f"sim_{prefix}_pick_origin",
            origin_dual=True,
        )
        with st.expander("Como interpretar"):
            st.markdown(
                f"- **Sim. origem**: cosseno entre mapas {sim.ORIGIN_GRID_COLS}×{sim.ORIGIN_GRID_ROWS} "
                "de onde os passes completos começam\n"
                "- **Sim. métricas**: percentil das métricas (Opção A) só entre os candidatos de origem parecida\n"
                "- **Origem dominante**: zona com maior % de passes daquele jogador"
            )


def main() -> None:
    classification_model = FIXED_CLASSIFICATION_MODEL
    tier_model = FIXED_TIER_MODEL
    xt_surface_mode = FIXED_XT_SURFACE_MODE

    with st.spinner("Carregando dados…"):
        _, all_players = load_analytics(
            tier_model=tier_model,
            classification_model=classification_model,
            xt_surface_mode=xt_surface_mode,
        )
        passes_by_player = load_passes(
            tier_model=tier_model,
            classification_model=classification_model,
            xt_surface_mode=xt_surface_mode,
        )
        serie_a_passes = load_serie_a_passes()

    rated, players_by_id, pool_by_position = compute_pass_ratings(all_players)
    selected_player_id = st.session_state.get("map_player_id")

    tab_pres, tab_dashboard, tab_sim_ba, tab_sim_ab = st.tabs(
        ["Apresentação", "Dashboard", "Similaridade B->A", "Similaridade A->B"]
    )
    with tab_pres:
        render_presentation_tab(all_players, passes_by_player)
    with tab_dashboard:
        render_map_section(all_players, players_by_id, pool_by_position, passes_by_player)
        st.divider()
        render_rating_section(rated, selected_player_id=selected_player_id)
    with tab_sim_ba:
        render_similarity_section(
            all_players, passes_by_player, serie_a_passes, sb_to_sa=True
        )
    with tab_sim_ab:
        render_similarity_section(
            all_players, passes_by_player, serie_a_passes, sb_to_sa=False
        )


if __name__ == "__main__":
    main()
