# Brasileirão 2026 — Passes xT

App Streamlit para análise de passes do Campeonato Brasileiro 2026 com métricas de xT heurístico v4.

## Dados

- `season_all_serieb.csv` — passes com coordenadas (Wyscout/SofaScore)
- `player_match_stats.csv` — minutos jogados por partida (filtro de elegibilidade no ranking)

### Extrair passes do SofaScore

```bash
pip install -r requirements-sofascore.txt

# Série A, Série B ou outra competição — use a URL com #id: da temporada
python -u scripts/onlypasses.py \
  --url "https://www.sofascore.com/football/tournament/brazil/brasileirao-serie-a/325#id:87678" \
  --output-dir "./passbr2026" \
  --consolidated-only \
  --resume \
  --rate-limit 1.0

# Copiar para a raiz do app (substitui season_all_serieb.csv)
python -u scripts/onlypasses.py ... \
  --copy-season-to-root \
  --season-filename season_all_serieb.csv
```

O CSV gerado inclui `position_raw` (código SofaScore) e `position` (LB, CAM, RW, … resolvida por escalação + coordenadas).

Arquivos auxiliares no `--output-dir`: `done_passes.json` (resume), `metadata_passes.json`, `failed_passes.json`.

## Executar

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Abas

- **Análise** — mapas de passes e destino por jogador
- **Ranking** — top 15 por grupo de posição (jogadores com ≥35% dos minutos do time)
