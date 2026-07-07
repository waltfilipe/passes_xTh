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
    fmt_count,
    fmt_metric_value,
    fmt_pct,
    load_passes_grouped,
    metric_label,
)
from passes_maps import draw_impact_pass_map

st.set_page_config(page_title="Passes xTh", layout="wide")

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.25rem; max-width: 1180px; }
    .player-card {
        background: linear-gradient(160deg, #151b2b 0%, #101522 100%);
        border: 1px solid #2a3550;
        border-radius: 12px;
        padding: 1rem 1.1rem;
        margin-top: 0.2rem;
    }
    .player-card h3 { margin: 0 0 0.15rem 0; color: #f1f5f9; font-size: 1.15rem; }
    .player-card .sub { color: #94a3b8; font-size: 0.85rem; margin-bottom: 0.75rem; }
    .player-card .rating {
        display: inline-block;
        font-size: 1.65rem;
        font-weight: 700;
        color: #7dd3fc;
        margin-bottom: 0.75rem;
    }
    .metric-line {
        display: flex;
        justify-content: space-between;
        gap: 0.75rem;
        padding: 0.28rem 0;
        border-bottom: 1px solid #1f293f;
        font-size: 0.8rem;
        color: #cbd5e1;
    }
    .metric-line span:last-child { color: #e2e8f0; font-weight: 600; white-space: nowrap; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Passes xTh — Série B")

RATING_COLUMNS = ["Jogador", "Time", "Rating"]
TOOLTIP_KEYS = (*TOOLTIP_EXTRA_KEYS, *RATING_METRIC_KEYS)


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
        return "#60a5fa"
    if rank == total:
        return "#f87171"
    t = (rank - 1) / (total - 1)
    r = int(52 + (248 - 52) * t)
    g = int(211 + (113 - 211) * t)
    b = int(153 + (113 - 153) * t)
    return f"#{r:02x}{g:02x}{b:02x}"


def _player_options(rated: list[dict]) -> list[tuple[str, str, str, str]]:
    rows = sorted(
        {(p["player_id"], p["player_name"], p.get("team", "—")) for p in rated},
        key=lambda x: _norm(x[1]),
    )
    return [(pid, name, team, f"{name} ({team})") for pid, name, team in rows]


def _sync_selection_from_query(rated_by_id: dict[str, dict]) -> None:
    qp = st.query_params.get("player_id")
    if qp and qp in rated_by_id:
        st.session_state["map_player_id"] = qp


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
        ranks = row.get("metric_ranks") if isinstance(row.get("metric_ranks"), dict) else {}
        tip = _tooltip_items(ranks)
        sel = " sel" if selected_player_id and str(row["player_id"]) == str(selected_player_id) else ""
        body.append(
            f'<tr class="row{sel}" data-pid="{pid}" onclick="pickPlayer(\'{pid}\')">'
            f"<td>{html.escape(str(row['Jogador']))}</td>"
            f"<td class='team'>{html.escape(str(row['Time']))}</td>"
            f'<td><div class="tip" onclick="event.stopPropagation()">'
            f'<span class="rating">{rating:.3f}</span>'
            f'<div class="tipbox">{tip}</div></div></td>'
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
.tip{{position:relative;display:inline-block}}
.rating{{font-weight:700;cursor:help;border-bottom:1px dashed #5b7aa5;color:#dbeafe}}
.tipbox{{display:none;position:absolute;z-index:9999;right:0;top:calc(100% + 8px);min-width:300px;
  background:#111827;border:1px solid #3d4f6f;border-radius:10px;padding:10px 12px;
  box-shadow:0 12px 30px rgba(0,0,0,.45)}}
.tip:hover .tipbox{{display:block}}
.tip-row{{display:flex;justify-content:space-between;gap:12px;padding:4px 0;border-bottom:1px solid #1f2937;font-size:0.78rem}}
.tip-row:last-child{{border-bottom:none}}
.tip-label{{color:#94a3b8}}
.tip-val{{font-weight:600;white-space:nowrap}}
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


def render_player_card(player: dict) -> None:
    pct = player.get("minutes_pct")
    pct_txt = fmt_pct(pct * 100.0) if pct is not None else "—"
    metrics_html = []
    for key in RATING_METRIC_KEYS:
        label = metric_label(key)
        value = fmt_metric_value(key, player.get(key))
        metrics_html.append(
            f'<div class="metric-line"><span>{html.escape(label)}</span><span>{html.escape(value)}</span></div>'
        )
    card = (
        '<div class="player-card">'
        f"<h3>{html.escape(player['player_name'])}</h3>"
        f'<div class="sub">{html.escape(player.get("team", "—"))} · {html.escape(str(player.get("position", "—")))}</div>'
        f'<div class="rating">{player.get("pass_rating", 0):.3f}</div>'
        f'<div class="metric-line"><span>Minutos</span><span>{html.escape(fmt_count(player.get("minutes")))}</span></div>'
        f'<div class="metric-line"><span>Passes</span><span>{html.escape(fmt_count(player.get("passes_completed")))}</span></div>'
        f'<div class="metric-line"><span>Min %</span><span>{html.escape(pct_txt)}</span></div>'
        + "".join(metrics_html)
        + "</div>"
    )
    st.markdown(card, unsafe_allow_html=True)


def render_map_section(
    rated: list[dict],
    rated_by_id: dict[str, dict],
    passes_by_player: dict,
) -> None:
    st.subheader("Mapa — passes de impacto")
    st.caption("Clique em um jogador na tabela de rating ou selecione abaixo.")

    options = _player_options(rated)
    if not options:
        st.info("Nenhum jogador elegível para o mapa.")
        return

    labels = [o[3] for o in options]
    id_by_label = {o[3]: o[0] for o in options}
    label_by_id = {o[0]: o[3] for o in options}

    _sync_selection_from_query(rated_by_id)
    map_player_id = st.session_state.get("map_player_id")
    default_index = None
    if map_player_id and map_player_id in label_by_id:
        default_index = labels.index(label_by_id[map_player_id])

    selected_label = st.selectbox(
        "Jogador",
        options=labels,
        index=default_index,
        placeholder="Selecione um jogador",
    )

    if selected_label:
        st.session_state["map_player_id"] = id_by_label[selected_label]
    else:
        st.session_state.pop("map_player_id", None)
        st.info("Selecione um jogador na lista ou clique em uma linha da tabela de rating.")
        return

    player_id = st.session_state["map_player_id"]
    player = rated_by_id[player_id]
    passes = passes_by_player.get(player_id)

    col_map, col_info = st.columns([1.35, 1.0], gap="large")
    with col_map:
        if passes is None or passes.empty:
            st.warning("Sem passes de impacto para este jogador.")
        else:
            fig = draw_impact_pass_map(passes, player["player_name"], player.get("team", "—"))
            st.pyplot(fig, clear_figure=True, use_container_width=False)
    with col_info:
        render_player_card(player)


def render_rating_section(rated: list[dict], *, selected_player_id: str | None) -> None:
    st.subheader("Rating por posição")
    st.caption(
        "Rating = média dos scores por métrica (1º = 1,0 · último = 0,5). "
        "Elegível: >30% dos jogos do time. Clique na linha para ver o mapa; passe o mouse no rating para rankings."
    )
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
                table_key=_norm(group),
            )


def main() -> None:
    with st.spinner("Carregando dados…"):
        _, players = load_analytics()
        passes_by_player = load_passes()

    rated = compute_pass_ratings(players)
    rated_by_id = {p["player_id"]: p for p in rated}
    selected_player_id = st.session_state.get("map_player_id")

    render_map_section(rated, rated_by_id, passes_by_player)
    st.divider()
    render_rating_section(rated, selected_player_id=selected_player_id)


if __name__ == "__main__":
    main()
