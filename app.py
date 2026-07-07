"""Passes xTh — Série B: rating por posição e mapa de passes de impacto."""

from __future__ import annotations

import html
import sys
import unicodedata
from pathlib import Path

_APP_ROOT = Path(__file__).resolve().parent
for _path in (_APP_ROOT, _APP_ROOT / "scripts"):
    _entry = str(_path)
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

import streamlit as st
import streamlit.components.v1 as components

import passes_engine as pe
from comparison_config import (
    COMPARISON_CARD_GROUPS,
    COMPARISON_IMPACT_KEYS,
    COMPARISON_PROGRESSION_KEYS,
)
from passes_maps import draw_impact_pass_map, draw_pass_destination_heatmap

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
build_analytics = pe.build_analytics
compute_pass_ratings = pe.compute_pass_ratings
compute_comparison_ratings = getattr(pe, "compute_comparison_ratings", None)
fmt_pct = pe.fmt_pct
fmt_stat_value = pe.fmt_stat_value
load_passes_grouped = pe.load_passes_grouped
metric_label = pe.metric_label
rank_to_display_score = pe.rank_to_display_score
score_display_color = pe.score_display_color
rate_player_vs_eligible_pool = pe.rate_player_vs_eligible_pool
rate_comparison_player_vs_pool = getattr(pe, "rate_comparison_player_vs_pool", None)


def fmt_rating_score(pass_rating) -> str:
    if pass_rating is None:
        return "—"
    return f"{float(pass_rating) * 10.0:.1f}"

st.set_page_config(page_title="Passes xTh", layout="wide")

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
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Passes xTh — Série B")

RATING_COLUMNS = ["Jogador", "Time", "Rating"]
SELECTBOX_KEY = "map_player_select"
COMPARISON_SELECT_KEY = "comparison_player_select"


@st.cache_data(show_spinner=False)
def load_analytics(_cache_version: int = DATA_CACHE_VERSION):
    return build_analytics(_cache_version)


@st.cache_data(show_spinner=False)
def load_passes(_cache_version: int = DATA_CACHE_VERSION):
    return load_passes_grouped(_cache_version)


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


def _comparison_card_header_html(title: str, card: dict) -> str:
    score = card.get("card_rating")
    pill = ""
    if score is not None:
        txt = fmt_rating_score(score)
        rank_info = card.get("card_rank")
        if rank_info:
            color = rank_color(int(rank_info["rank"]), int(rank_info["total"]))
            txt_color = _badge_text_color(color)
            rank_txt = f'{int(rank_info["rank"])}/{int(rank_info["total"])}'
            if card.get("rating_is_compared"):
                rank_txt += " · vs aptos"
            elif card.get("rating_is_solo"):
                rank_txt += " · individual"
            pill = (
                f'<span class="section-rating-tip">'
                f'<span class="section-rating-pill" style="background:{color};color:{txt_color}">'
                f"{html.escape(txt)}</span>"
                f'<span class="rating-tipbox">{html.escape(rank_txt)}</span>'
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


def _comparison_card_html(player: dict, section_key: str, title: str, keys: tuple[str, ...]) -> str:
    cards = player.get("comparison_cards") if isinstance(player.get("comparison_cards"), dict) else {}
    card = cards.get(section_key, {})
    metric_ranks = card.get("metric_ranks") if isinstance(card.get("metric_ranks"), dict) else {}
    parts = [_comparison_card_header_html(title, card)]
    for key in keys:
        parts.append(
            _metric_line_html(
                metric_label(key),
                key,
                _stat_display(player, key),
                metric_ranks,
                show_rank=True,
            )
        )
    return '<div class="player-card">' + "".join(parts) + "</div>"


def _resolve_comparison_player(
    player: dict,
    comparison_pool_by_group: dict[str, list[dict]],
) -> dict:
    resolved = dict(player)
    if resolved.get("eligible_for_rating"):
        return resolved
    if rate_comparison_player_vs_pool is None:
        return resolved
    group = str(resolved.get("position_group") or "—")
    pool = comparison_pool_by_group.get(group, [])
    cards: dict[str, dict] = {}
    for section_key, keys in COMPARISON_CARD_GROUPS.items():
        cards[section_key] = rate_comparison_player_vs_pool(resolved, pool, section_key, keys)
    resolved["comparison_cards"] = cards
    return resolved


def render_comparison_section(
    all_players: list[dict],
    comparison_players_by_id: dict[str, dict],
    comparison_pool_by_group: dict[str, list[dict]],
) -> None:
    st.subheader("Comparação por perfil de passe")
    st.caption(
        "Rating por card = média das notas das métricas do card no grupo de posição "
        "(1º = 9,0 · mediano = 6,0 · último = 3,0). "
        "Passes progressivos seguem a regra Wyscout."
    )

    options = _player_options(all_players)
    if not options:
        st.info("Nenhum jogador disponível para comparação.")
        return

    labels = [o[3] for o in options]
    id_by_label = {o[3]: o[0] for o in options}

    selected_labels = st.multiselect(
        "Jogadores",
        options=labels,
        key=COMPARISON_SELECT_KEY,
        placeholder="Selecione um ou mais jogadores",
    )
    if not selected_labels:
        st.info("Selecione ao menos um jogador para ver os cards de comparação.")
        return

    for label in selected_labels:
        player_id = id_by_label[label]
        player = _resolve_comparison_player(
            dict(comparison_players_by_id.get(player_id, {})),
            comparison_pool_by_group,
        )
        if not player:
            continue

        st.markdown(
            f"### {html.escape(player.get('player_name', '—'))} "
            f"<span style='color:#94a3b8;font-size:0.95rem;font-weight:500'>"
            f"{html.escape(str(player.get('team', '—')))} · "
            f"{html.escape(str(player.get('position', '—')))} · "
            f"{html.escape(str(player.get('position_group', '—')))}"
            f"</span>",
            unsafe_allow_html=True,
        )
        col_impact, col_progression = st.columns(2, gap="small")
        with col_impact:
            st.markdown(
                _comparison_card_html(
                    player,
                    "comparison_impact",
                    "Impacto & Agressão",
                    COMPARISON_IMPACT_KEYS,
                ),
                unsafe_allow_html=True,
            )
        with col_progression:
            st.markdown(
                _comparison_card_html(
                    player,
                    "comparison_progression",
                    "Progressão & Criação",
                    COMPARISON_PROGRESSION_KEYS,
                ),
                unsafe_allow_html=True,
            )
        st.divider()


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


def main() -> None:
    with st.spinner("Carregando dados…"):
        _, all_players = load_analytics()
        passes_by_player = load_passes()

    rated, players_by_id, pool_by_position = compute_pass_ratings(all_players)
    selected_player_id = st.session_state.get("map_player_id")

    tab_dashboard, tab_comparison = st.tabs(["Dashboard", "Comparação"])
    with tab_dashboard:
        render_map_section(all_players, players_by_id, pool_by_position, passes_by_player)
        st.divider()
        render_rating_section(rated, selected_player_id=selected_player_id)
    with tab_comparison:
        if compute_comparison_ratings is None:
            st.error(
                "A aba Comparação precisa da versão mais recente de passes_engine.py. "
                "Reimplante o app no Streamlit Cloud (ou reinicie o serviço) para carregar o commit mais recente."
            )
        else:
            comparison_players_by_id, comparison_pool_by_group = compute_comparison_ratings(all_players)
            render_comparison_section(
                all_players,
                comparison_players_by_id,
                comparison_pool_by_group,
            )


if __name__ == "__main__":
    main()
