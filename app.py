"""Passes xTh — Série B: rating por posição e mapa de passes de impacto."""

from __future__ import annotations

import html
import unicodedata

import streamlit as st
import streamlit.components.v1 as components

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


TOOLTIP_KEYS = (*TOOLTIP_EXTRA_KEYS, *RATING_METRIC_KEYS)


def _tooltip_items(metric_ranks: dict) -> str:
    items = []
    for key in TOOLTIP_KEYS:
        info = metric_ranks.get(key)
        if not info:
            continue
        rank = int(info["rank"])
        total = int(info["total"])
        label = TOOLTIP_LABELS.get(key, key)
        color = rank_color(rank, total)
        items.append(
            '<div class="tip-row">'
            f'<span class="tip-label">{html.escape(label)}</span>'
            f'<span class="tip-val" style="color:{color}">{rank}/{total}</span>'
            "</div>"
        )
    return "".join(items)


def render_rating_table(rows: list[dict]) -> None:
    if not rows:
        st.info("Nenhum jogador elegível nesta posição.")
        return

    body = []
    for row in rows:
        rating = float(row["Rating"])
        ranks = row.get("metric_ranks") if isinstance(row.get("metric_ranks"), dict) else {}
        tip = _tooltip_items(ranks)
        body.append(
            "<tr>"
            f"<td>{html.escape(str(row['Jogador']))}</td>"
            f"<td>{html.escape(str(row['Time']))}</td>"
            f'<td><div class="tip"><span class="rating">{rating:.3f}</span>'
            f'<div class="tipbox">{tip}</div></div></td>'
            "</tr>"
        )

    page = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><style>
*{{box-sizing:border-box}}
body{{margin:0;padding:0;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;color:#0f172a;background:#fff}}
.rx{{width:100%;border-collapse:collapse;font-size:0.92rem}}
.rx th,.rx td{{padding:9px 12px;border-bottom:1px solid #e2e8f0;text-align:left;vertical-align:middle}}
.rx th{{background:#f8fafc;font-weight:600;color:#334155}}
.rx tr:hover td{{background:#f8fafc}}
.tip{{position:relative;display:inline-block}}
.rating{{font-weight:700;cursor:help;border-bottom:1px dashed #94a3b8}}
.tipbox{{display:none;position:absolute;z-index:9999;right:0;top:calc(100% + 8px);min-width:320px;
  background:#fff;border:1px solid #cbd5e1;border-radius:8px;padding:10px 12px;
  box-shadow:0 10px 28px rgba(15,23,42,.18)}}
.tip:hover .tipbox{{display:block}}
.tip-row{{display:flex;justify-content:space-between;gap:12px;padding:4px 0;border-bottom:1px solid #f1f5f9;font-size:0.8rem}}
.tip-row:last-child{{border-bottom:none}}
.tip-label{{color:#475569}}
.tip-val{{font-weight:600;white-space:nowrap}}
</style></head><body>
<table class="rx"><thead><tr>
{"".join(f"<th>{html.escape(c)}</th>" for c in RATING_COLUMNS)}
</tr></thead><tbody>{"".join(body)}</tbody></table>
</body></html>"""

    height = min(44 * len(rows) + 52, 920)
    components.html(page, height=height, scrolling=False)


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
            render_rating_table(rows)


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
