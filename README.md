# GTA VI Trophy Rarity Forecast — v2

Forecasts each Grand Theft Auto VI (PS5) trophy's earn rate on Exophase at
2026-11-26T00:00:00Z, for the make.fun competition
`gta6-trophy-rarity-forecast-01`.

![make.fun competition](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)

`main.py` is the entrypoint: `python3 main.py` reads one case file (`<id>.in`)
from `MAKEFUN_INPUT_DIR` (default `/input`) and writes `<id>.out` to
`MAKEFUN_OUTPUT_DIR` (default `/output`). Pure standard library, no network.
Runtime is well under the 8-second limit (~0.2–0.4 s per case locally).

## Design (v2)

All forecasting is on the logit scale. Consecutive reads of an unchanged
page collapse to a single snapshot.

**Trend regime (≥ 2 distinct snapshots).** Two median components, combined
in logit space:

- *Direct trend:* per-trophy weighted least squares of `logit(rate)` on
  `ln(1 + hours)` with 24-hour half-life weights. The slope is shrunk toward
  the **tier median** (platinum / gold / silver / bronze) with adaptive
  weight `n / (n + 2)` from the number of distinct snapshots; the intercept
  is refit at the shrunk slope. Extrapolation is damped toward the last
  read (`damp = 1 / (1 + days / 5)`).
- *Structural:* earners and players-tracked are forecast separately and the
  rate is `earners / tracked`. `ln(tracked)` is the mean of three growth
  forms in log space — power-law, exponential on the last 24 h of distinct
  observations, and saturating-exponential (grid search on `k`).
  `ln(earners)` per trophy is a WLS fit with the same adaptive tier-median
  shrinkage as the direct trend.

Ensemble median: `0.35 * structural + 0.65 * direct trend` (logit space).

**Spread:** variance = residual × leverage + random walk (`0.5² × days`);
the refresh-jump accumulation term is removed (jump scale 0). Spread =
`0.8 × sqrt(variance)`, clipped to `[0.35, 2.2]` logit (tuned in the submit
session: grid 0.6/0.8/1.0 gave pooled loss 0.4199/0.4197/0.4355, all with
coverage 0.982; 0.8 chosen for the best pooled loss).

**Flat regime (1 distinct snapshot):** unchanged from v1 — a dilution
model. The tracked count jumps ×6.06 (the one recorded refresh, 638 →
3,868 players) and the new cohort earns at 0.10× the old rate, so the
median is the last rate × 0.2485, with a wide spread (~1 logit). This is
why the practice forecasts (p00/p01) are identical to v1's.

**100% trophy:** a trophy last seen at 100% is hedged at a 99.85% median
(spread 0.9 logit in trend regime, ~1 logit in flat regime).

## Backtest (run in the build session, 2026-11-21)

`eval/backtest.py` scores v1 (`eval/v1_ref.py`, re-implemented from its
design description) against v2 on a 12-scenario **cohort simulator** that is
deliberately different from both forecasters: player counts follow a
saturating curve calibrated on the one recorded refresh (638 → 3,868);
cohorts earn toward a per-trophy eventual fraction with a casual discount
(0.35) for later joiners; the page the forecaster sees only updates at
Exophase refreshes. History 2.5 days, horizon 4.5 days (6,510 min).
Scenarios: 3 player-growth speeds × 3 progression speeds, plus 3
adversarial cases (player surge day 4, progression stall after day 3,
refresh storm). Scoring follows the competition rules exactly (pinball loss
on the logit scale, typical miss, 90% coverage).

| scenario | v1 loss | v2 loss |
|---|---:|---:|
| growth-slow / prog-slow | 0.5472 | 0.4388 |
| growth-slow / prog-med | 0.5110 | 0.3579 |
| growth-slow / prog-fast | 0.4451 | 0.2936 |
| growth-med / prog-slow | 0.5591 | 0.5737 |
| growth-med / prog-med | 0.5267 | 0.4831 |
| growth-med / prog-fast | 0.4677 | 0.3444 |
| growth-fast / prog-slow | 0.5367 | 0.6005 |
| growth-fast / prog-med | 0.5107 | 0.5264 |
| growth-fast / prog-fast | 0.4616 | 0.3893 |
| adversarial: surge day 4 | 0.4773 | 0.3530 |
| adversarial: stall day 3 | 0.3663 | 0.2419 |
| adversarial: refresh storm | 0.5031 | 0.4336 |

Pooled over all 12 scenarios: **v1 loss 0.4927, v2 loss 0.4197 (−14.8%)**;
v2 wins **9/12** scenarios; typical miss v1 1.986 vs v2 1.877 points.
Coverage 90%: v1 0.982, v2 0.982. Practice cases (answers estimated by the
dilution model, f = 0.10 and 0.30): v1 and v2 losses identical (0.2331 /
0.3343), as designed.

Full per-scenario table and assumptions: `eval/results.md`.
Verification (`eval/verify.py`): 15/15 checks pass — practice cases valid
(55 rows, 5 non-decreasing quantiles, ≤ 65,536 bytes), byte-identical
across two runs, runtime < 8 s, and edge cases pass (1 read, null rates,
missing `players_tracked`, empty history, 2-snapshot trend regime).

## Files

- `main.py` — the forecast program (entrypoint).
- `eval/v1_ref.py` — v1 baseline reference, for comparison only.
- `eval/backtest.py` — 12-scenario cohort-simulator backtest.
- `eval/verify.py` — grader-style verification.
- `eval/results.md` — backtest numbers from the build session.
