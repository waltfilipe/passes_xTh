# Brasileirão 2026 — Passes xT

App Streamlit para análise de passes do Campeonato Brasileiro 2026 com métricas de xT heurístico v4.

## Dados

- `season_all_br.csv` — passes com coordenadas (Wyscout/SofaScore)
- `player_match_stats.csv` — minutos jogados por partida (filtro de elegibilidade no ranking)

## Executar

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Abas

- **Análise** — mapas de passes e destino por jogador
- **Ranking** — top 15 por grupo de posição (jogadores com ≥35% dos minutos do time)
