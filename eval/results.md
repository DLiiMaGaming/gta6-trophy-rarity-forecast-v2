# Backtest v1 vs v2 — cohort simulator (12 scenarios)

Run in this session (fresh code, fresh simulator). Simulator: cohort model calibrated on the one recorded refresh (638 -> 3,868 players at 04:30 launch night); history 2.5 days, horizon 4.5 days (6,510 min).

## Calibration check

first-refresh rate ratio vs launch: min 0.404, max 0.413 (all < 1: True)

## Per-scenario pinball loss (logit scale)

| scenario | v1 loss | v2 loss | v1 cov | v2 cov | v1 miss | v2 miss |
|---|---:|---:|---:|---:|---:|---:|
| growth-slow/prog-slow | 0.5472 | 0.4388 | 0.982 | 0.982 | 3.862 | 3.371 |
| growth-slow/prog-med | 0.5110 | 0.3579 | 0.982 | 0.982 | 2.657 | 2.216 |
| growth-slow/prog-fast | 0.4451 | 0.2936 | 0.982 | 0.982 | 1.112 | 0.770 |
| growth-med/prog-slow | 0.5591 | 0.5737 | 0.982 | 0.982 | 3.284 | 3.212 |
| growth-med/prog-med | 0.5267 | 0.4831 | 0.982 | 0.982 | 2.389 | 2.327 |
| growth-med/prog-fast | 0.4677 | 0.3444 | 0.982 | 0.982 | 1.166 | 1.122 |
| growth-fast/prog-slow | 0.5367 | 0.6005 | 0.982 | 0.982 | 2.409 | 2.561 |
| growth-fast/prog-med | 0.5107 | 0.5264 | 0.982 | 0.982 | 1.814 | 1.959 |
| growth-fast/prog-fast | 0.4616 | 0.3893 | 0.982 | 0.982 | 0.964 | 1.092 |
| ADVERSARIAL surge-day4 | 0.4773 | 0.3530 | 0.982 | 0.982 | 1.514 | 1.452 |
| ADVERSARIAL stall-day3 | 0.3663 | 0.2419 | 0.982 | 0.982 | 0.158 | 0.096 |
| ADVERSARIAL refresh-storm | 0.5031 | 0.4336 | 0.982 | 0.982 | 2.509 | 2.351 |

## Pooled (all 12 scenarios, 55 trophies x 5 quantiles each)

- v1 loss pooled: **0.4927**
- v2 loss pooled: **0.4197** (-14.8% vs v1)
- v2 wins **9/12** scenarios
- coverage 90%: v1 0.982, v2 0.982
- typical miss (points): v1 1.986, v2 1.877

## Practice cases (answers estimated by dilution model)

No reference answers in the kit; answer estimated as last rate x (1 + f*(6.06-1))/6.06. v2 must equal v1 here (flat regime unchanged).

- p00/f=0.1: v1 loss 0.2331, v2 loss 0.2331 (identical: True)
- p00/f=0.3: v1 loss 0.3343, v2 loss 0.3343 (identical: True)
- p01/f=0.1: v1 loss 0.2331, v2 loss 0.2331 (identical: True)
- p01/f=0.3: v1 loss 0.3343, v2 loss 0.3343 (identical: True)

## Assumptions recorded (v1 reconstruction)

- v1 jump-refresh variance = 1.0 logit^2 (one refresh-sized shock); v1 spread unclipped; v1 100%-trophy spread 1.0.
- v2 spread = 0.8 * sqrt(resid*leverage + 0.25*days), clipped to [0.35, 2.2]; jump term removed (scale 0). Tuned in the submit session: grid 0.6/0.8/1.0 gave pooled loss 0.4199/0.4197/0.4355 with coverage 0.982 in all three; 0.8 chosen (best pooled loss).
- v2 structural tracked count: mean of three log-space predictions (power-law, exponential on last 24 h, saturating-exponential by grid search on k).
