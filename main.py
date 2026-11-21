#!/usr/bin/env python3
"""Forecast GTA VI (PS5) trophy earn rates on Exophase -- v2.

Fresh implementation. Design (see README.md):
  * logit scale; consecutive identical reads collapse to one snapshot;
  * trend regime (>= 2 distinct snapshots): per-trophy WLS of logit(rate)
    on ln(1 + hours) with 24 h half-life weights, slope shrunk toward the
    TIER median with adaptive weight n / (n + 2), extrapolation damped
    toward the last read; plus a structural component that forecasts
    earners and players-tracked separately (three growth forms for the
    tracked count, averaged in log space) and combines them;
    ensemble median = 0.35 * structural + 0.65 * direct trend (logit);
    spread from residual * leverage + random walk, x0.8, clipped
    to [0.35, 2.2] logit; the refresh-jump accumulation term is removed;
  * flat regime (1 distinct snapshot): dilution model -- the tracked
    count jumps x6.06 (the one recorded refresh, 638 -> 3,868) and the
    new cohort earns at 0.10x the old rate; wide spread ~1 logit;
  * a trophy last seen at 100% is hedged at a 99.85% median, spread 0.9.

Reads one case file (<id>.in) from MAKEFUN_INPUT_DIR (default /input)
and writes <id>.out to MAKEFUN_OUTPUT_DIR (default /output).
Pure standard library; no network.
"""

import glob
import json
import math
import os
import sys
from datetime import datetime, timezone

Z_TAU = (-1.6448536269514722, -0.6744897501960817, 0.0,
         0.6744897501960817, 1.6448536269514722)
HALF_LIFE_H = 24.0
DILUTION_GROWTH = 6.06      # players 638 -> 3,868 at the recorded refresh
DILUTION_EARN = 0.10        # new cohort earns at 0.10x the old rate
FLAT_SPREAD = 1.0
HEDGE_MEDIAN = 99.85
HEDGE_SPREAD = 0.9
STRUCT_W = 0.35            # ensemble weight of the structural component
SPREAD_MULT = 0.8
SPREAD_FLOOR = 0.35
SPREAD_CAP = 2.2
RW_SIGMA = 0.5             # logit random-walk sigma per sqrt(day)
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
    """Weighted least squares y ~ a + b*x. Returns (a, b, xbar, sxx, sigma2)."""
    sw = sum(ws)
    if sw <= 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    xbar = sum(w * x for w, x in zip(ws, xs)) / sw
    ybar = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - xbar) ** 2 for w, x in zip(ws, xs))
    sxy = sum(w * (x - xbar) * (y - ybar) for w, x, y in zip(ws, xs, ys))
    if sxx <= 1e-12:
        b = 0.0
    else:
        b = sxy / sxx
    a = ybar - b * xbar
    resid2 = sum(w * (y - a - b * x) ** 2 for w, x, y in zip(ws, xs, ys)) / sw
    sigma = max(math.sqrt(max(resid2, 0.0)), SIGMA_FLOOR)
    return a, b, xbar, sxx, sigma


def shrink_slope(b_own, b_target, n):
    w_own = n / (n + 2.0)
    return w_own * b_own + (1.0 - w_own) * b_target


def refit_intercept(xs, ys, ws, b):
    sw = sum(ws)
    if sw <= 0:
        return 0.0
    return sum(w * (y - b * x) for w, x, y in zip(ws, xs, ys)) / sw


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
    """Collapse consecutive reads with identical players + rates."""
    snaps = []
    prev_key = None
    for read in history or []:
        rates = read.get("rates") or []
        key = (read.get("players_tracked"),
               tuple(rates))
        if key == prev_key:
            continue
        prev_key = key
        snaps.append(read)
    return snaps


def forecast(case):
    trophies = case.get("trophies") or []
    history = case.get("history") or []
    target_t = parse_time(case.get("target"))
    horizon_min = case.get("horizon_minutes")
    if target_t is None and horizon_min is not None and history:
        last_t = parse_time(history[-1].get("at")) or parse_time(history[-1].get("slot"))
        if last_t is not None:
            from datetime import timedelta
            target_t = last_t + timedelta(minutes=float(horizon_min))

    snaps = distinct_snapshots(history)
    n = len(snaps)

    # Times in hours since the first snapshot.
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
    days = 0.0
    if target_t is not None:
        days = max(0.0, (target_t - last_t).total_seconds() / 86400.0)
    x_target = math.log1p(max(0.0, (target_t - t0).total_seconds() / 3600.0)) \
        if target_t is not None else 0.0

    # Last known players-tracked and last non-null rate per trophy.
    n_last = None
    for s in reversed(snaps):
        if s.get("players_tracked") is not None:
            n_last = float(s["players_tracked"])
            break
    n_trophies = len(trophies)

    def last_rate(i):
        for s in reversed(snaps):
            rates = s.get("rates") or []
            if i < len(rates) and rates[i] is not None:
                return float(rates[i])
        return None

    last_rates = [last_rate(i) for i in range(n_trophies)]
    known = [r for r in last_rates if r is not None]
    tier_fallback = median(known) if known else 1.0

    rows = []
    if n <= 1:
        # ---- Flat regime: dilution model (unchanged from v1). ----
        if n_last and n_last > 0:
            factor = (1.0 + DILUTION_EARN * (DILUTION_GROWTH - 1.0)) / DILUTION_GROWTH
        else:
            factor = 1.0
        for i, tr in enumerate(trophies):
            r = last_rates[i]
            if r is None:
                r = tier_fallback
            if r >= RATE_HI:
                # Flat regime is unchanged from v1, including the hedge
                # spread (~1 logit); the v2 0.9 spread applies in trend.
                rows.append({"id": tr["id"],
                             "q": quantiles(logit(HEDGE_MEDIAN), FLAT_SPREAD)})
                continue
            med = clamp(r * factor, RATE_LO, RATE_HI)
            rows.append({"id": tr["id"],
                         "q": quantiles(logit(med), FLAT_SPREAD)})
        return {"trophies": rows}

    # ---- Trend regime: build per-trophy series. ----
    xs_all = []
    ws_all = []
    t_last_h = hours_of(snaps[-1])
    for s in snaps:
        h = hours_of(s)
        xs_all.append(math.log1p(max(0.0, h)))
        ws_all.append(0.5 ** max(0.0, (t_last_h - h) / HALF_LIFE_H))

    def series(i, transform):
        xs, ys, ws = [], [], []
        for k, s in enumerate(snaps):
            rates = s.get("rates") or []
            if i >= len(rates) or rates[i] is None:
                continue
            y = transform(float(rates[i]), s)
            if y is None:
                continue
            xs.append(xs_all[k])
            ys.append(y)
            ws.append(ws_all[k])
        return xs, ys, ws

    # Direct trend fits: logit(rate) ~ ln(1+hours).
    direct = []   # per trophy: (a, b_raw, xbar, sxx, sigma) or None
    for i in range(n_trophies):
        xs, ys, ws = series(i, lambda r, s: logit(r))
        if len(xs) >= 2:
            direct.append(wls(xs, ys, ws))
        else:
            direct.append(None)

    def tier_median_slope(fits, tiers):
        by_tier = {}
        for i, f in enumerate(fits):
            if f is None:
                continue
            by_tier.setdefault(tiers[i], []).append(f[1])
        med = {t: median(v) for t, v in by_tier.items()}
        all_slopes = [f[1] for f in fits if f is not None]
        return med, median(all_slopes) if all_slopes else 0.0

    tiers = [tr.get("tier", "bronze") for tr in trophies]
    tier_med_d, glob_med_d = tier_median_slope(direct, tiers)

    # Structural: earners fits ln(E) ~ ln(1+hours), E = rate * N / 100.
    earners = []
    for i in range(n_trophies):
        def _ln_e(r, s, i=i):
            n_j = s.get("players_tracked")
            if n_j is None or float(n_j) <= 0:
                return None
            e = max(r, RATE_LO) * float(n_j) / 100.0
            return math.log(max(e, 1e-9))
        xs, ys, ws = series(i, _ln_e)
        if len(xs) >= 2:
            earners.append(wls(xs, ys, ws))
        else:
            earners.append(None)
    tier_med_e, glob_med_e = tier_median_slope(earners, tiers)

    # Tracked-count growth: three forms, averaged in log space.
    def forecast_tracked():
        pts = []
        for k, s in enumerate(snaps):
            nv = s.get("players_tracked")
            if nv is None or float(nv) <= 0:
                continue
            pts.append((hours_of(s), xs_all[k], math.log(float(nv)), ws_all[k]))
        if not pts:
            return None
        if len(pts) == 1 or n_last is None:
            return n_last
        preds = []
        # (a) power-law: ln(N) ~ ln(1+hours)
        a, b, _, _, _ = wls([p[1] for p in pts], [p[2] for p in pts],
                            [p[3] for p in pts])
        preds.append(a + b * x_target)
        # (b) exponential on the last 24 h of distinct observations,
        # anchored at the last distinct snapshot. With fewer than two
        # distinct times in the window the form is undefined and skipped
        # (it must not fall back to all history: that would extrapolate
        # the launch-night explosion).
        t_last_distinct = pts[-1][0]
        recent = [p for p in pts if p[0] >= t_last_distinct - 24.0]
        if len({p[0] for p in recent}) >= 2:
            ts = [p[0] for p in recent]
            ls = [p[2] for p in recent]
            mt, ml = sum(ts) / len(ts), sum(ls) / len(ls)
            sxx = sum((t - mt) ** 2 for t in ts)
            if sxx > 1e-12:
                be = sum((t - mt) * (l - ml) for t, l in zip(ts, ls)) / sxx
                ae = ml - be * mt
                t_tgt = (target_t - t0).total_seconds() / 3600.0 if target_t else t_last_h
                preds.append(ae + be * t_tgt)
        # (c) saturating exponential N = A - B*exp(-k t), grid search on k
        best = None
        k = 0.002
        while k <= 1.0:
            us = [math.exp(-k * p[0]) for p in pts]
            ns = [math.exp(p[2]) for p in pts]
            mu, mn = sum(us) / len(us), sum(ns) / len(ns)
            suu = sum((u - mu) ** 2 for u in us)
            if suu > 1e-12:
                c = sum((u - mu) * (nn - mn) for u, nn in zip(us, ns)) / suu
                A = mn - c * mu
                if c < 0 and A > max(ns):
                    sse = sum((nn - (A + c * u)) ** 2 for u, nn in zip(us, ns))
                    if best is None or sse < best[0]:
                        best = (sse, A, c, k)
            k *= 1.18
        if best is not None:
            _, A, c, k = best
            t_tgt = (target_t - t0).total_seconds() / 3600.0 if target_t else t_last_h
            n_pred = A + c * math.exp(-k * t_tgt)
            if n_pred > 0:
                preds.append(math.log(n_pred))
        if not preds:
            return n_last
        lo = math.log(max(n_last * 0.9, 1.0))
        hi = math.log(n_last * 50.0)
        preds = [clamp(p, lo, hi) for p in preds]
        return math.exp(sum(preds) / len(preds))

    n_target = forecast_tracked()
    damp = 1.0 / (1.0 + days / 5.0)

    for i, tr in enumerate(trophies):
        r = last_rates[i]
        if r is None:
            r = tier_fallback
        if r >= RATE_HI:
            rows.append({"id": tr["id"],
                         "q": quantiles(logit(HEDGE_MEDIAN), HEDGE_SPREAD)})
            continue
        d = direct[i]
        if d is None:
            # Too few points for this trophy: fall back to dilution.
            factor = (1.0 + DILUTION_EARN * (DILUTION_GROWTH - 1.0)) / DILUTION_GROWTH \
                if n_last else 1.0
            med = clamp(r * factor, RATE_LO, RATE_HI)
            rows.append({"id": tr["id"],
                         "q": quantiles(logit(med), FLAT_SPREAD)})
            continue
        a, b_raw, xbar, sxx, sigma = d
        b_med = tier_med_d.get(tiers[i], glob_med_d)
        b = shrink_slope(b_raw, b_med, n)
        # Refit intercept with the shrunk slope on this trophy's series.
        xs_i, ys_i, ws_i = series(i, lambda rr, s: logit(rr))
        a2 = refit_intercept(xs_i, ys_i, ws_i, b)
        y_direct = a2 + b * x_target
        y_direct = logit(r) + damp * (y_direct - logit(r))

        # Structural component.
        y_struct = None
        e = earners[i]
        if e is not None and n_target:
            ae, be_raw, _, _, _ = e
            be_med = tier_med_e.get(tiers[i], glob_med_e)
            be = shrink_slope(be_raw, be_med, n)
            # Recompute the earners series for the intercept refit.
            xs_e, ys_e, ws_e = [], [], []
            for k, s in enumerate(snaps):
                rates = s.get("rates") or []
                nv = s.get("players_tracked")
                if i >= len(rates) or rates[i] is None or nv is None \
                        or float(nv) <= 0:
                    continue
                xs_e.append(xs_all[k])
                ys_e.append(math.log(max(float(rates[i]), RATE_LO)
                                     * float(nv) / 100.0))
                ws_e.append(ws_all[k])
            ae2 = refit_intercept(xs_e, ys_e, ws_e, be)
            ln_e_tgt = ae2 + be * x_target
            e_tgt = math.exp(clamp(ln_e_tgt, math.log(1e-9),
                                   math.log(max(n_target, 1.0))))
            r_struct = clamp(100.0 * e_tgt / n_target, RATE_LO, RATE_HI)
            y_struct = logit(r_struct)

        if y_struct is None:
            y_med = y_direct
        else:
            y_med = STRUCT_W * y_struct + (1.0 - STRUCT_W) * y_direct
        y_med = clamp(y_med, logit(RATE_LO), logit(RATE_HI))

        # Spread: residual * leverage + random walk, x0.4, clipped.
        lev = 1.0
        if sxx > 1e-12:
            lev = 1.0 + (x_target - xbar) ** 2 / sxx
        var = (sigma ** 2) * lev + (RW_SIGMA ** 2) * days
        spread = clamp(SPREAD_MULT * math.sqrt(max(var, 0.0)),
                       SPREAD_FLOOR, SPREAD_CAP)
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
