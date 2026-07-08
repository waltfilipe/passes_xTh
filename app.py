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
POSITION_GROUPS_ORDER = pe.POSITION_GROUPS_ORDER
RATING_TOP_N = pe.RATING_TOP_N
RATING_MIN_MINUTES_PCT = pe.RATING_MIN_MINUTES_PCT
RATING_MIN_PASSES_PCT = pe.RATING_MIN_PASSES_PCT
SIMILARITY_TOP_K = 10
SIMILARITY_DIRECTION_KEY = "similarity_direction"
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
rank_to_display_score = pe.rank_to_display_score
score_display_color = pe.score_display_color
rate_player_vs_eligible_pool = pe.rate_player_vs_eligible_pool


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
    .rating-warning-tip:hover .rating-tipbox {
        display: block;
    }
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


def _metric_line_html(
    label: str,
    key: str,
    value: str,
    metric_ranks: dict,
    *,
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
    value_html = (
        f'<span class="val-wrap">{badge}<span class="stat-val">{html.escape(value)}</span></span>'
        if badge
        else f'<span class="stat-val">{html.escape(value)}</span>'
    )
    return (
        '<div class="metric-line">'
        f"<span>{html.escape(label)}</span>"
        f"{value_html}"
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
                    metric_label(key),
                    key,
                    _stat_display(player, key),
                    metric_ranks,
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


def _player_card_html(
    player: dict,
    sections: list[tuple[str, str | None, tuple[str, ...], bool]],
) -> str:
    metric_ranks = player.get("metric_ranks") if isinstance(player.get("metric_ranks"), dict) else {}
    return (
        '<div class="player-card">'
        + _build_sections_html(player, metric_ranks, sections)
        + "</div>"
    )


def render_player_layout(player: dict, passes) -> None:
    team_label = player.get("team", "—")
    col_map1, col_map2 = st.columns(2, gap="small")

    if passes is None or passes.empty:
        with col_map1:
            st.warning("Sem passes de impacto para este jogador.")
    else:
        with col_map1:
            fig = draw_impact_pass_map(passes, player["player_name"], team_label, compact=False)
            st.pyplot(fig, clear_figure=True, use_container_width=True)
        with col_map2:
            fig_heat = draw_pass_destination_heatmap(passes, player["player_name"], team_label, compact=False)
            st.pyplot(fig_heat, clear_figure=True, use_container_width=True)

    general_sections: list[tuple[str, str | None, tuple[str, ...], bool]] = [
        (
            "Geral",
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
    abs_rel_sections: list[tuple[str, str | None, tuple[str, ...], bool]] = [
        ("Métricas Absolutas", "metrics_absolute", ABSOLUTE_METRIC_KEYS, True),
        ("Métricas Relativas", "metrics_relative", RELATIVE_METRIC_KEYS, True),
    ]
    long_ball_sections: list[tuple[str, str | None, tuple[str, ...], bool]] = [
        ("Long balls", "long_balls", LONG_BALL_STAT_KEYS, True),
    ]
    style_sections: list[tuple[str, str | None, tuple[str, ...], bool]] = [
        ("Construção", "construction", CONSTRUCTION_METRIC_KEYS, True),
        ("Agressão", "aggression", AGGRESSION_METRIC_KEYS, True),
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

    col_general, col_metrics, col_long, col_style = st.columns(4, gap="small")
    with col_general:
        st.markdown(general_card, unsafe_allow_html=True)
    with col_metrics:
        st.markdown(_player_card_html(player, abs_rel_sections), unsafe_allow_html=True)
    with col_long:
        st.markdown(_player_card_html(player, long_ball_sections), unsafe_allow_html=True)
    with col_style:
        st.markdown(_player_card_html(player, style_sections), unsafe_allow_html=True)


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


def _render_similarity_target_card(
    player: dict,
    passes,
    *,
    league: str,
    pool_label: str,
    pool_size: int,
) -> None:
    name = html.escape(str(player.get("player_name", "—")))
    team = html.escape(str(player.get("team", "—")))
    position = html.escape(str(player.get("position", "—")))
    group = html.escape(str(player.get("position_group", "—")))
    minutes_txt = fmt_stat_value("minutes", player.get("minutes"))
    passes_txt = fmt_stat_value("passes_completed", player.get("passes_completed"))

    col_info, col_map = st.columns([1.0, 1.15], gap="medium")
    with col_info:
        st.markdown(
            f"### {name}\n"
            f"{team} · {position} · {group} · **{html.escape(league)}**",
            unsafe_allow_html=True,
        )
        m1, m2 = st.columns(2)
        m1.metric("Minutos", minutes_txt)
        m2.metric("Passes", passes_txt)
        profile = sim.pass_origin_profile(passes) if passes is not None else None
        if profile is not None:
            origin_txt = sim.describe_dominant_origin_zone(profile)
            st.caption(f"Origem dominante: {origin_txt}")
        st.caption(f"Pool de busca: **{html.escape(pool_label)}** ({pool_size} jogadores)")

    with col_map:
        if passes is not None and not passes.empty:
            fig = draw_pass_origin_heatmap(
                passes,
                str(player.get("player_name", "—")),
                str(player.get("team", "—")),
                mini=True,
            )
            st.pyplot(fig, clear_figure=True, use_container_width=True)
        else:
            st.info("Sem passes para mapa de origem.")


def render_similarity_section(
    all_players: list[dict],
    passes_by_player_sb: dict,
    serie_a_passes: dict,
) -> None:
    import pandas as pd

    st.subheader("Similaridade entre ligas")
    st.caption(
        f"Top {SIMILARITY_TOP_K} jogadores mais parecidos na mesma posição detalhada "
        f"(LB, RB, CM, LW…). Fonte Série A: season_all_brfull.csv com position_raw."
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

    direction = st.radio(
        "Direção da busca",
        options=("Série B → Série A", "Série A → Série B"),
        horizontal=True,
        key=SIMILARITY_DIRECTION_KEY,
    )
    sb_to_sa = direction == "Série B → Série A"

    serie_a_by_pos = sim.group_players_by_detailed_position(serie_a_players)
    sb_by_pos = sim.group_players_by_detailed_position(all_players)
    players_sb_by_id = {str(p["player_id"]): p for p in all_players}
    players_sa_by_id = {str(p["player_id"]): p for p in serie_a_players}

    if sb_to_sa:
        options = _player_options(all_players)
        select_label = "Jogador Série B"
        select_key = SIMILARITY_SELECT_SB_KEY
    else:
        options = _player_options(serie_a_players)
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
        pool_passes = serie_a_passes
        pool_label = f"Série A · {search_pos or '—'}"
        target_league = "Série B"
    else:
        target = dict(players_sa_by_id[target_id])
        target_passes = serie_a_passes.get(target_id)
        pool = sim.similarity_search_pool(sb_by_pos, search_pos)
        pool_passes = passes_by_player_sb
        pool_label = f"Série B · {search_pos or '—'}"
        target_league = "Série A"

    if not search_pos:
        st.warning("Posição inválida para comparação (goleiros são excluídos).")
        return

    if not pool:
        st.warning(
            f"Nenhum jogador elegível na posição **{html.escape(search_pos)}** em {pool_label.split(' · ')[0]}."
        )
        return

    _render_similarity_target_card(
        target,
        target_passes,
        league=target_league,
        pool_label=pool_label,
        pool_size=len(pool),
    )
    st.divider()

    tab_a, tab_c, tab_origin = st.tabs(
        ["Opção A — percentil", "Opção C — z-score", "Origem dos passes"]
    )

    def _results_df(results: list[dict], *, include_origin: bool = False) -> pd.DataFrame:
        rows = []
        for rank, r in enumerate(results, start=1):
            row = {
                "#": rank,
                "Similaridade": f"{r.get('similarity_pct', 0):.1f}%",
                "Jogador": r.get("player_name", "—"),
                "Time": r.get("team", "—"),
                "Posição": r.get("position", "—"),
            }
            if include_origin:
                row["Origem dominante"] = r.get("origin_dominant", "—")
            else:
                row["Impact p90"] = fmt_stat_value("impact_passes_p90", r.get("impact_passes_p90"))
                row["PHI p90"] = fmt_stat_value("phi_p90", r.get("phi_p90"))
                row["ΔxT / pass"] = fmt_stat_value("dxt_per_pass", r.get("dxt_per_pass"))
                row["Prog. p90"] = fmt_stat_value(
                    "progressive_passes_p90", r.get("progressive_passes_p90")
                )
            rows.append(row)
        return pd.DataFrame(rows)

    top_k = SIMILARITY_TOP_K
    dest_league = "Série A" if sb_to_sa else "Série B"

    with tab_a:
        st.caption(
            f"Distância euclidiana no perfil percentil (0–100) dentro do pool {dest_league}."
        )
        results_a = sim.find_similar_option_a(target, pool, top_k=top_k)
        if not results_a:
            st.info("Nenhum similar encontrado.")
        else:
            st.dataframe(_results_df(results_a), use_container_width=True, hide_index=True)
            with st.expander("Métricas usadas (Opção A)"):
                st.write(", ".join(metric_label(k) for k in sim.SIMILARITY_METRICS_A))

    with tab_c:
        st.caption(
            f"Distância euclidiana ponderada em z-scores do pool {dest_league} "
            "(maior peso em impact p90, PHI p90 e ΔxT p90)."
        )
        results_c = sim.find_similar_option_c(target, pool, top_k=top_k)
        if not results_c:
            st.info("Nenhum similar encontrado.")
        else:
            st.dataframe(_results_df(results_c), use_container_width=True, hide_index=True)
            with st.expander("Métricas usadas (Opção C)"):
                st.write(", ".join(metric_label(k) for k in sim.SIMILARITY_METRICS_A))

    with tab_origin:
        st.caption(
            "Similaridade da região de origem dos passes completos: grid 12×8, "
            "proporção por zona e distância por cosseno entre perfis."
        )
        target_profile = sim.pass_origin_profile(target_passes)
        if target_profile is None:
            st.warning(
                "Sem passes completos suficientes para montar o perfil de origem deste jogador."
            )
            return

        if not pool_passes:
            st.warning("Passes do pool indisponíveis para comparação espacial.")
            return

        results_origin = sim.find_similar_option_origin(
            target_passes,
            pool,
            pool_passes,
            top_k=top_k,
        )
        if not results_origin:
            st.info("Nenhum similar por origem de passe encontrado no pool.")
        else:
            st.dataframe(
                _results_df(results_origin, include_origin=True),
                use_container_width=True,
                hide_index=True,
            )
            with st.expander("Como interpretar"):
                st.markdown(
                    "- **defesa (área)**: maior parte dos passes sai de dentro ou na entrada da área própria\n"
                    "- **saída de bola**: origem predominante entre a área e o meio defensivo\n"
                    "- **meio-campo** / **terço final**: origem mais avançada no campo\n"
                    "- Esquerda / centro / direita: corredor lateral do campo (vista de cima)"
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

    tab_dashboard, tab_similarity = st.tabs(["Dashboard", "Similaridade"])
    with tab_dashboard:
        render_map_section(all_players, players_by_id, pool_by_position, passes_by_player)
        st.divider()
        render_rating_section(rated, selected_player_id=selected_player_id)
    with tab_similarity:
        render_similarity_section(all_players, passes_by_player, serie_a_passes)


if __name__ == "__main__":
    main()
