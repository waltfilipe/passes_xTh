"""Série B — Passes xT: Rating e Ranking."""

import pandas as pd
import streamlit as st

from passes_engine import (
    DATA_CACHE_VERSION,
    GROUP_COLORS,
    POSITION_GROUPS_ORDER,
    RANKING_METRIC_GROUPS,
    RANKING_TOP_N,
    RATING_METRIC_KEYS,
    RATING_MIN_MINUTES_PCT,
    RATING_SCORE_BEST,
    RATING_SCORE_WORST,
    RATING_TOP_N,
    SEASON_ALL_CSV_PATH,
    PLAYER_MATCH_STATS_PATH,
    build_analytics,
    compute_pass_ratings,
    fmt_count,
    fmt_decimal,
    fmt_metric_value,
    fmt_pct,
    metric_label,
)

st.set_page_config(layout="wide", page_title="Série B — Passes xT")

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.5rem; }
    .player-header {
        font-size: 1.15rem;
        font-weight: 700;
        color: #eef1f7;
        margin-bottom: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="Carregando season_all_serieb.csv…")
def load_analytics(_cache_version: int = DATA_CACHE_VERSION):
    return build_analytics(_cache_version)


def _ranking_table(players: list[dict], metric_key: str, metric_label_text: str) -> pd.DataFrame:
    ranked = sorted(players, key=lambda p: p.get(metric_key, 0) or 0, reverse=True)[:RANKING_TOP_N]
    rows = []
    for idx, player in enumerate(ranked, start=1):
        rows.append({
            "#": idx,
            "Jogador": player["player_name"],
            "Pos.": player.get("position", "—"),
            "Time": player.get("team", "—"),
            metric_label_text: fmt_metric_value(metric_key, player.get(metric_key)),
            "Min": fmt_count(player.get("minutes")),
            "Passes": fmt_count(player.get("passes_completed")),
        })
    return pd.DataFrame(rows)


def render_rating_tab(players: list[dict]) -> None:
    st.markdown("### Rating · Passes")
    st.caption(
        f"Jogadores de linha com mais de **{int(RATING_MIN_MINUTES_PCT * 100)}%** dos minutos do time. "
        f"Por métrica e posição: **1º = {RATING_SCORE_BEST:.1f}**, **último = {RATING_SCORE_WORST:.1f}**. "
        f"Rating = média de **{len(RATING_METRIC_KEYS)}** scores."
    )
    if not players:
        st.warning("Nenhum jogador elegível para o rating.")
        return

    rated = compute_pass_ratings(players)
    st.caption(f"{len(rated)} elegíveis · top **{RATING_TOP_N}** por posição")

    with st.expander("Métricas do rating", expanded=False):
        for title, keys in RANKING_METRIC_GROUPS:
            labels = ", ".join(metric_label(k) for k in keys)
            st.markdown(f"**{title}** — {labels}")

    by_position: dict[str, list[dict]] = {}
    for player in rated:
        by_position.setdefault(str(player.get("position") or "—"), []).append(player)

    for position in sorted(by_position.keys()):
        top = sorted(by_position[position], key=lambda p: p.get("pass_rating", 0), reverse=True)[:RATING_TOP_N]
        st.markdown(
            f'<div class="player-header" style="border-left:4px solid #5b9bd5;padding-left:0.5rem;">'
            f"{position}</div>",
            unsafe_allow_html=True,
        )
        rows = []
        for idx, player in enumerate(top, start=1):
            pct = player.get("minutes_pct")
            rows.append({
                "#": idx,
                "Jogador": player["player_name"],
                "Time": player.get("team", "—"),
                "Rating": fmt_decimal(player.get("pass_rating")),
                "Min": fmt_count(player.get("minutes")),
                "Min %": fmt_pct(pct * 100.0) if pct is not None else "—",
                "Passes": fmt_count(player.get("passes_completed")),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.markdown("---")


def render_ranking_tab(players: list[dict]) -> None:
    st.markdown("### Ranking · Passes")
    st.caption(
        f"Jogadores elegíveis (≥{int(RATING_MIN_MINUTES_PCT * 100)}% min) · "
        f"top **{RANKING_TOP_N}** por métrica e grupo de posição."
    )
    if not players:
        st.warning("Nenhum jogador elegível.")
        return

    by_group: dict[str, list[dict]] = {g: [] for g in POSITION_GROUPS_ORDER}
    for player in players:
        group = player.get("position_group")
        if group in by_group:
            by_group[group].append(player)

    ranking_metrics = [
        (group_title, key, metric_label(key))
        for group_title, keys in RANKING_METRIC_GROUPS
        for key in keys
    ]

    for group in POSITION_GROUPS_ORDER:
        group_players = by_group[group]
        if not group_players:
            continue
        color = GROUP_COLORS.get(group, "#94a3b8")
        st.markdown(
            f'<div class="player-header" style="border-left:4px solid {color};padding-left:0.5rem;">'
            f"{group}</div>",
            unsafe_allow_html=True,
        )
        current_group: str | None = None
        cols = st.columns(3)
        slot = 0
        for group_title, metric_key, label in ranking_metrics:
            if group_title != current_group:
                current_group = group_title
                st.markdown(f"**{group_title}**")
                cols = st.columns(3)
                slot = 0
            with cols[slot % 3]:
                st.markdown(f"**{label}**")
                st.dataframe(
                    _ranking_table(group_players, metric_key, label),
                    use_container_width=True,
                    hide_index=True,
                    height=420,
                )
            slot += 1
            if slot % 3 == 0:
                cols = st.columns(3)
        st.markdown("---")


# ── MAIN ────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style="text-align:center;margin-bottom:1rem;">
      <h1 style="margin:0;color:#eef1f7;">Série B — Passes xT</h1>
      <p style="color:#94a3b8;font-size:0.95rem;margin-top:0.35rem;">
        Rating composto · Rankings por métrica · xT Heurístico v4
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

registry, players = load_analytics()
if not registry or not players:
    st.error(f"Dataset não encontrado ou sem jogadores elegíveis. Coloque `{SEASON_ALL_CSV_PATH.name}` na raiz.")
    st.stop()

with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center;">
          <h3 style="margin:0;color:#eef1f7;">Passes xT</h3>
          <p style="color:#94a3b8;font-size:0.85rem;">Série B</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(f"{len(registry)} jogadores · {len(players)} elegíveis · xT v4")
    with st.expander("Arquivos", expanded=False):
        st.markdown(f"✓ `{SEASON_ALL_CSV_PATH.name}`")
        st.markdown(f"{'✓' if PLAYER_MATCH_STATS_PATH.exists() else '✗'} `{PLAYER_MATCH_STATS_PATH.name}`")

tab_rating, tab_ranking = st.tabs(["Rating", "Ranking"])

with tab_rating:
    render_rating_tab(players)

with tab_ranking:
    render_ranking_tab(players)
