"""Passes xTh — Série B: rating por posição e mapa de passes de impacto."""

from __future__ import annotations

import html
import unicodedata

import pandas as pd
import streamlit as st

from passes_engine import (
    POSITION_GROUPS_ORDER,
    RATING_METRIC_KEYS,
    RATING_TOP_N,
    TOOLTIP_EXTRA_KEYS,
    TOOLTIP_LABELS,
    build_analytics,
    compute_pass_ratings,
    load_passes_grouped,
)
from passes_maps import draw_impact_pass_map

st.set_page_config(page_title="Passes xTh", layout="wide")
st.title("Passes xTh — Série B")

RATING_COLUMNS = ["Jogador", "Time", "Rating"]


@st.cache_data(show_spinner=False)
def load_analytics():
    return build_analytics()


@st.cache_data(show_spinner=False)
def load_passes():
    return load_passes_grouped()


def _norm(s: str) -> str:
    return unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()


def rank_color(rank: int, total: int) -> str:
    if total <= 1:
        return "#94a3b8"
    if rank <= 5:
        return "#3b82f6"
    if rank == total:
        return "#ef4444"
    t = (rank - 1) / (total - 1)
    r = int(34 + (239 - 34) * t)
    g = int(197 + (68 - 197) * t)
    b = int(94 + (68 - 94) * t)
    return f"#{r:02x}{g:02x}{b:02x}"


def _tooltip_rows(metric_ranks: dict) -> str:
    rows = []
    for key in TOOLTIP_EXTRA_KEYS + list(RATING_METRIC_KEYS):
        info = metric_ranks.get(key)
        if not info:
            continue
        rank = int(info["rank"])
        total = int(info["total"])
        label = TOOLTIP_LABELS.get(key, key)
        color = rank_color(rank, total)
        rows.append(
            f'<tr><td>{html.escape(label)}</td>'
            f'<td style="color:{color};font-weight:600">{rank}/{total}</td></tr>'
        )
    return "\n".join(rows)


def render_rating_table(df: pd.DataFrame) -> None:
    if df.empty:
        st.info("Nenhum jogador elegível nesta posição.")
        return

    css = """
    <style>
    .rx{width:100%;border-collapse:collapse;font-size:0.9rem}
    .rx th,.rx td{padding:8px 10px;border-bottom:1px solid #e2e8f0;text-align:left}
    .rx th{background:#f8fafc;font-weight:600}
    .rx tr:hover td{background:#f8fafc}
    .tip{position:relative;display:inline-block;cursor:help}
    .tip>span{font-weight:700;color:#0f172a}
    .tipbox{display:none;position:absolute;z-index:1000;right:0;top:calc(100% + 6px);
      min-width:300px;background:#fff;border:1px solid #cbd5e1;border-radius:8px;
      padding:10px;box-shadow:0 8px 24px rgba(15,23,42,.15)}
    .tip:hover .tipbox{display:block}
    .tipbox table{width:100%;border-collapse:collapse;font-size:0.78rem}
    .tipbox td{padding:3px 6px;border-bottom:1px solid #f1f5f9}
    .tipbox td:last-child{text-align:right;white-space:nowrap}
    </style>
    """
    body = []
    for _, row in df.iterrows():
        rating = float(row["Rating"])
        tip = _tooltip_rows(row.get("metric_ranks") or {})
        body.append(
            "<tr>"
            f"<td>{html.escape(str(row['Jogador']))}</td>"
            f"<td>{html.escape(str(row['Time']))}</td>"
            f'<td><div class="tip"><span>{rating:.3f}</span>'
            f'<div class="tipbox"><table>{tip}</table></div></div></td>'
            "</tr>"
        )
    table = (
        f"{css}<table class='rx'><thead><tr>"
        + "".join(f"<th>{c}</th>" for c in RATING_COLUMNS)
        + "</tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table>"
    )
    st.markdown(table, unsafe_allow_html=True)


def render_rating_section(players: list[dict]) -> None:
    st.subheader("Rating por posição")
    st.caption(
        "Rating = média dos scores por métrica (1º = 1,0 · último = 0,5). "
        "Elegível: >30% dos jogos do time. Passe o mouse no rating para ver rankings."
    )
    rated = compute_pass_ratings(players)
    for group in POSITION_GROUPS_ORDER:
        subset = sorted(
            [p for p in rated if p["position_group"] == group],
            key=lambda p: p.get("pass_rating", 0),
            reverse=True,
        )[:RATING_TOP_N]
        if not subset:
            continue
        with st.expander(f"{group} ({len(subset)})", expanded=group in {"Goleiro", "Zagueiro"}):
            rows = [
                {
                    "Jogador": p["player_name"],
                    "Time": p["team"],
                    "Rating": p["pass_rating"],
                    "metric_ranks": p.get("metric_ranks", {}),
                }
                for p in subset
            ]
            render_rating_table(pd.DataFrame(rows))


def render_impact_map(players: list[dict], passes_by_player: dict) -> None:
    st.subheader("Mapa — passes de impacto")
    st.caption("Passes progressivos e de alto impacto concluídos (layout clássico).")

    options = sorted(
        {(p["player_id"], p["player_name"], p["team"]) for p in players},
        key=lambda x: _norm(x[1]),
    )
    if not options:
        st.info("Nenhum jogador elegível para o mapa.")
        return

    labels = [f"{name} ({team})" for _, name, team in options]
    label = st.selectbox("Jogador", labels, key="impact_map_player")
    idx = labels.index(label)
    player_id, player_name, team = options[idx]

    passes = passes_by_player.get(player_id)
    if passes is None or passes.empty:
        st.warning("Sem passes de impacto para este jogador.")
        return

    fig = draw_impact_pass_map(passes, player_name, team)
    st.pyplot(fig, clear_figure=True)


def main() -> None:
    with st.spinner("Carregando dados…"):
        _, players = load_analytics()
        passes_by_player = load_passes()

    render_rating_section(players)
    st.divider()
    render_impact_map(players, passes_by_player)


if __name__ == "__main__":
    main()
