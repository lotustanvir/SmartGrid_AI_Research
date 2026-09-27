# EXP_TEMPLATE — copy, never edit in place

See `docs/EXPERIMENT_GUIDE.md`. Duplicate this folder per EXP-ID, then fill:

- `config.yaml` — dataset/feature/model/hypers/seed/splits record (§2).
- `metrics.json` — full metric set per horizon/slice (written by runner).
- `verification.json` — 5/5 leakage checks (written by runner).
- `README.md` — one-paragraph purpose + link to registry row + paper table (if any).
