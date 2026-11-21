#!/usr/bin/env python3
"""v1 baseline reference (re-implemented fresh from its design description).

  * logit scale; consecutive identical reads collapse to one snapshot;
  * trend regime (>= 2 distinct snapshots): per-trophy WLS of logit(rate)
    on ln(1 + hours) with 24 h half-life weights; slope shrunk FIXED
    0.6 own / 0.4 GLOBAL median; extrapolation damped toward the last
    read (damp = 1 / (1 + days / 5)); spread = sqrt(resid * leverage
    + random-walk variance (0.5^2 * days) + 1x jump-refresh variance
    (1.0 logit^2, one refresh-sized shock));
  * flat regime (1 distinct snapshot): dilution model -- players x6.06,
    new cohort earns at 0.10x the old rate; spread ~1 logit;
  * a trophy last seen at 100% is hedged at a 99.85% median.

Same CLI as main.py (MAKEFUN_INPUT_DIR / MAKEFUN_OUTPUT_DIR).
Pure standard library.
"""

import glob
import json
import math
import os
import sys
from datetime import datetime, timedelta, timezone

Z_TAU = (-1.6448536269514722, -0.6744897501960817, 0.0,
         0.6744897501960817, 1.6448536269514722)
HALF_LIFE_H = 24.0
DILUTION_GROWTH = 6.06
DILUTION_EARN = 0.10
FLAT_SPREAD = 1.0
HEDGE_MEDIAN = 99.85
HEDGE_SPREAD = 1.0
JUMP_VAR = 1.0
RW_SIGMA = 0.5
SIGMA_FLOOR = 0.05
RATE_LO = 0.01
RATE_HI = 99.99


def parse_time(s):
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def logit(rate_pct):
    p = clamp(rate_pct, RATE_LO, RATE_HI) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(y):
    y = clamp(y, logit(RATE_LO), logit(RATE_HI))
    return 100.0 / (1.0 + math.exp(-y))


def median(vals):
    v = sorted(vals)
    n = len(v)
    if n == 0:
        return 0.0
    return v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])


def wls(xs, ys, ws):
    sw = sum(ws)
    if sw <= 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    xbar = sum(w * x for w, x in zip(ws, xs)) / sw
    ybar = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - xbar) ** 2 for w, x in zip(ws, xs))
    sxy = sum(w * (x - xbar) * (y - ybar) for w, x, y in zip(ws, xs, ys))
    b = sxy / sxx if sxx > 1e-12 else 0.0
    a = ybar - b * xbar
    resid2 = sum(w * (y - a - b * x) ** 2 for w, x, y in zip(ws, xs, ys)) / sw
    sigma = max(math.sqrt(max(resid2, 0.0)), SIGMA_FLOOR)
    return a, b, xbar, sxx, sigma


def quantiles(y_med, spread):
    out = []
    prev = RATE_LO
    for z in Z_TAU:
        q = round(inv_logit(y_med + z * spread), 4)
        q = clamp(max(q, prev), RATE_LO, RATE_HI)
        out.append(q)
        prev = q
    return out


def distinct_snapshots(history):
    snaps = []
    prev_key = None
    for read in history or []:
        key = (read.get("players_tracked"), tuple(read.get("rates") or []))
        if key == prev_key:
            continue
        prev_key = key
        snaps.append(read)
    return snaps


def forecast(case):
    trophies = case.get("trophies") or []
    history = case.get("history") or []
    target_t = parse_time(case.get("target"))
    if target_t is None and case.get("horizon_minutes") is not None and history:
        last_t = parse_time(history[-1].get("at")) or parse_time(history[-1].get("slot"))
        if last_t is not None:
            target_t = last_t + timedelta(minutes=float(case["horizon_minutes"]))

    snaps = distinct_snapshots(history)
    n = len(snaps)

    t0 = None
    for s in snaps:
        t0 = parse_time(s.get("at")) or parse_time(s.get("slot"))
        if t0 is not None:
            break
    if t0 is None:
        t0 = target_t or datetime.now(timezone.utc)

    def hours_of(read):
        t = parse_time(read.get("at")) or parse_time(read.get("slot"))
        return 0.0 if t is None else (t - t0).total_seconds() / 3600.0

    last = snaps[-1] if snaps else {}
    last_t = parse_time(last.get("at")) or parse_time(last.get("slot")) or t0
    days = max(0.0, (target_t - last_t).total_seconds() / 86400.0) \
        if target_t is not None else 0.0
    x_target = math.log1p(max(0.0, (target_t - t0).total_seconds() / 3600.0)) \
        if target_t is not None else 0.0

    n_last = None
    for s in reversed(snaps):
        if s.get("players_tracked") is not None:
            n_last = float(s["players_tracked"])
            break

    def last_rate(i):
        for s in reversed(snaps):
            rates = s.get("rates") or []
            if i < len(rates) and rates[i] is not None:
                return float(rates[i])
        return None

    last_rates = [last_rate(i) for i in range(len(trophies))]
    known = [r for r in last_rates if r is not None]
    fallback = median(known) if known else 1.0
    factor = (1.0 + DILUTION_EARN * (DILUTION_GROWTH - 1.0)) / DILUTION_GROWTH \
        if n_last else 1.0

    rows = []
    if n <= 1:
        for i, tr in enumerate(trophies):
            r = last_rates[i]
            if r is None:
                r = fallback
            if r >= RATE_HI:
                rows.append({"id": tr["id"],
                             "q": quantiles(logit(HEDGE_MEDIAN), HEDGE_SPREAD)})
                continue
            rows.append({"id": tr["id"],
                         "q": quantiles(logit(clamp(r * factor, RATE_LO, RATE_HI)),
                                        FLAT_SPREAD)})
        return {"trophies": rows}

    xs_all, ws_all = [], []
    t_last_h = hours_of(snaps[-1])
    for s in snaps:
        h = hours_of(s)
        xs_all.append(math.log1p(max(0.0, h)))
        ws_all.append(0.5 ** max(0.0, (t_last_h - h) / HALF_LIFE_H))

    fits = []
    for i in range(len(trophies)):
        xs, ys, ws = [], [], []
        for k, s in enumerate(snaps):
            rates = s.get("rates") or []
            if i < len(rates) and rates[i] is not None:
                xs.append(xs_all[k])
                ys.append(logit(float(rates[i])))
                ws.append(ws_all[k])
        fits.append(wls(xs, ys, ws) if len(xs) >= 2 else None)
    glob_med = median([f[1] for f in fits if f is not None])
    damp = 1.0 / (1.0 + days / 5.0)

    for i, tr in enumerate(trophies):
        r = last_rates[i]
        if r is None:
            r = fallback
        if r >= RATE_HI:
            rows.append({"id": tr["id"],
                         "q": quantiles(logit(HEDGE_MEDIAN), HEDGE_SPREAD)})
            continue
        f = fits[i]
        if f is None:
            rows.append({"id": tr["id"],
                         "q": quantiles(logit(clamp(r * factor, RATE_LO, RATE_HI)),
                                        FLAT_SPREAD)})
            continue
        a, b_raw, xbar, sxx, sigma = f
        b = 0.6 * b_raw + 0.4 * glob_med
        xs, ys, ws = [], [], []
        for k, s in enumerate(snaps):
            rates = s.get("rates") or []
            if i < len(rates) and rates[i] is not None:
                xs.append(xs_all[k])
                ys.append(logit(float(rates[i])))
                ws.append(ws_all[k])
        sw = sum(ws)
        a2 = sum(w * (y - b * x) for w, x, y in zip(ws, xs, ys)) / sw if sw > 0 else a
        y_med = a2 + b * x_target
        y_med = logit(r) + damp * (y_med - logit(r))
        y_med = clamp(y_med, logit(RATE_LO), logit(RATE_HI))
        lev = 1.0 + (x_target - xbar) ** 2 / sxx if sxx > 1e-12 else 1.0
        var = (sigma ** 2) * lev + (RW_SIGMA ** 2) * days + JUMP_VAR
        spread = math.sqrt(max(var, 0.0))
        rows.append({"id": tr["id"], "q": quantiles(y_med, spread)})
    return {"trophies": rows}


def main(argv=None):
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    ins = sorted(glob.glob(os.path.join(in_dir, "*.in")))
    if not ins:
        raise SystemExit("no .in file in %s" % in_dir)
    path = ins[0]
    with open(path, encoding="utf-8") as f:
        case = json.load(f)
    out = forecast(case)
    os.makedirs(out_dir, exist_ok=True)
    cid = case.get("id") or os.path.splitext(os.path.basename(path))[0]
    with open(os.path.join(out_dir, cid + ".out"), "w", encoding="utf-8") as f:
        json.dump(out, f, separators=(",", ":"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
