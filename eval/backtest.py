#!/usr/bin/env python3
"""Backtest v1 vs v2 on a 12-scenario cohort simulator + practice estimates.

Simulator (deliberately different from both forecasters): a cohort model.
  * Time origin 2026-11-19T00:00:00Z. History: reads every 30 min from
    23:50 the night before to the checkpoint (21 Nov 11:30Z, t = 59.5 h);
    target t = 168 h (26 Nov 00:00Z), horizon 6,510 min.
  * Tracked players N(t) = N_inf - (N_inf - 638) * exp(-g t), with (N_inf, g)
    calibrated so N passes through the ONE recorded refresh: 638 -> 3,868
    players at t = 4.67 h. Three growth speeds (N_inf 12k / 30k / 100k).
  * Each trophy i: day-one cohort (638 players, mean age 6 h at t = 0)
    earns toward eventual fraction u_i = r_i0 / (1 - exp(-6 * lam_i));
    later cohorts are casual (factor 0.35) and earn with the same hazard
    lam_i = lam_speed * (0.3 + 0.7 * sqrt(r_i0 / 100)). Three progression
    speeds (lam 0.008 / 0.02 / 0.05 per hour).
  * The page the forecaster sees only updates at Exophase refreshes
    (first at 04:30, then every 24 h); between refreshes the displayed
    page is stale. The answer is the displayed page at the last refresh
    before the target.
  * Adversarial: (a) player surge on day 4 (tracked count doubles in one
    step, new cohort casual), (b) progression stall after day 3 (no new
    earning, players keep growing), (c) refresh storm (refreshes every
    6 h, so the history jumps often and the answer is nearly current).

Scoring follows RULES exactly: pinball loss on the logit scale, typical
miss (mean |y - median| in points), 90% coverage.
"""

import json
import math
import os
import sys
from datetime import datetime, timedelta, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
KIT = os.path.expanduser("~/workspace/makefun/gta6-trophy-rarity-forecast/kit")
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
import main as v2
import v1_ref as v1

BASE = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)
N0 = 638.0
REFRESH_T = 4.0 + 40.0 / 60.0          # 04:30 launch night -> 3868 players
REFRESH_N = 3868.0
CHECKPOINT_T = 59.5                    # 21 Nov 11:30Z
TARGET_T = 168.0                       # 26 Nov 00:00Z
CASUAL = 0.35
INIT_AGE_H = 6.0
TAUS = (0.05, 0.25, 0.5, 0.75, 0.95)


def iso(t_hours):
    return (BASE + timedelta(hours=t_hours)).strftime("%Y-%m-%dT%H:%M:%SZ")


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def logit(pct):
    p = clamp(pct, 0.01, 99.99) / 100.0
    return math.log(p / (1.0 - p))


def pinball(y, q, tau):
    d = logit(y) - logit(q)
    return 2.0 * max(tau * d, (tau - 1.0) * d)


def score(forecast_rows, answers):
    """forecast_rows: [{"id","q":[...]}]; answers: {id: rate}. Returns dict."""
    losses, misses, covered = [], [], 0
    by_id = {r["id"]: r["q"] for r in forecast_rows}
    for tid, y in answers.items():
        q = by_id[tid]
        for tau, qq in zip(TAUS, q):
            losses.append(pinball(y, qq, tau))
        misses.append(abs(y - q[2]))
        if q[0] <= y <= q[4]:
            covered += 1
    n = len(answers)
    return {"loss": sum(losses) / len(losses),
            "typical_miss": sum(misses) / n,
            "coverage": covered / n}


class CohortSim:
    def __init__(self, n_inf, lam_speed, surge=False, stall=False,
                 storm=False):
        self.n_inf = float(n_inf)
        self.lam_speed = float(lam_speed)
        self.surge = surge
        self.stall = stall
        self.storm = storm
        d = self.n_inf - N0
        self.g = -math.log((self.n_inf - REFRESH_N) / d) / REFRESH_T
        self.d = d
        if storm:
            self.refreshes = [REFRESH_T + 6.0 * k for k in range(10)]
            self.answer_refresh = 166.6667
        else:
            self.refreshes = [REFRESH_T, 28.6667, 52.6667]
            self.answer_refresh = 148.6667

    def n_true(self, t):
        n = self.n_inf - self.d * math.exp(-self.g * t)
        if self.surge and t >= 96.0:
            n += self.n_inf - self.d * math.exp(-self.g * 96.0)
        return n

    def _integral(self, t, lam):
        """int_0^t N'(s) (1 - exp(-lam (t-s))) ds, N'(s) = D g exp(-g s)."""
        if t <= 0:
            return 0.0
        g = self.g
        term1 = (1.0 - math.exp(-g * t)) / g
        if abs(lam - g) < 1e-9:
            term2 = t * math.exp(-lam * t)
        else:
            term2 = math.exp(-lam * t) * (math.exp((lam - g) * t) - 1.0) / (lam - g)
        return self.d * g * (term1 - term2)

    def earners_true(self, t, u, lam, r0):
        if r0 >= 99.99:
            return self.n_true(t)
        te = min(t, 72.0) if self.stall else t
        e = N0 * u * (1.0 - math.exp(-lam * (te + INIT_AGE_H)))
        e += CASUAL * u * self._integral(te, lam)
        if self.surge and t >= 96.0:
            n96 = self.n_inf - self.d * math.exp(-self.g * 96.0)
            e += n96 * CASUAL * u * (1.0 - math.exp(-lam * (t - 96.0)))
        return e

    def state(self, t, trophies, r0):
        """True (N, rates) at time t."""
        n = self.n_true(t)
        rates = []
        for i, tr in enumerate(trophies):
            lam = self.lam_speed * (0.3 + 0.7 * math.sqrt(r0[i] / 100.0))
            if r0[i] >= 99.99:
                u = 1.0
            else:
                u = min(r0[i] / 100.0 / (1.0 - math.exp(-6.0 * lam)), 0.999)
            e = self.earners_true(t, u, lam, r0[i])
            rates.append(round(100.0 * e / n, 2))
        return n, rates

    def displayed(self, t, trophies, r0):
        """Page as the forecaster sees it at read time t (stale refreshes)."""
        rt = None
        for r in self.refreshes:
            if r <= t:
                rt = r
        if rt is None:
            return N0, list(r0)
        return self.state(rt, trophies, r0)

    def build_case(self, trophies, r0):
        history = []
        t = -1.0 / 6.0
        while t <= CHECKPOINT_T + 1e-9:
            n, rates = self.displayed(t, trophies, r0)
            earned = int(round(sum(r / 100.0 * n for r in rates)))
            history.append({"slot": iso(max(t, -1.0 / 6.0)),
                            "at": iso(t),
                            "players_tracked": int(round(n)),
                            "trophies_earned": earned,
                            "rates": rates})
            t += 0.5
        ans_n, ans_rates = self.state(self.answer_refresh, trophies, r0)
        answers = {tr["id"]: ans_rates[i] for i, tr in enumerate(trophies)}
        case = {"schema": "makefun/gta6-trophies-case/v1",
                "id": "sim",
                "at": iso(CHECKPOINT_T),
                "horizon_minutes": 6510,
                "target": iso(TARGET_T),
                "levels": list(TAUS),
                "game": "Grand Theft Auto VI",
                "trophies": trophies,
                "history": history,
                "notes": "synthetic cohort-simulator case"}
        return case, answers


def run():
    with open(os.path.join(KIT, "input", "p00.in"), encoding="utf-8") as f:
        p00 = json.load(f)
    trophies = p00["trophies"]
    r0 = [float(x) for x in p00["history"][0]["rates"]]

    scenarios = []
    for gname, n_inf in (("slow", 12000.0), ("med", 30000.0), ("fast", 100000.0)):
        for pname, lam in (("slow", 0.008), ("med", 0.02), ("fast", 0.05)):
            scenarios.append((f"growth-{gname}/prog-{pname}",
                              CohortSim(n_inf, lam)))
    scenarios.append(("ADVERSARIAL surge-day4",
                      CohortSim(30000.0, 0.02, surge=True)))
    scenarios.append(("ADVERSARIAL stall-day3",
                      CohortSim(30000.0, 0.02, stall=True)))
    scenarios.append(("ADVERSARIAL refresh-storm",
                      CohortSim(30000.0, 0.02, storm=True)))

    # Calibration check: first refresh must lower every rate (except 100%).
    sim0 = CohortSim(30000.0, 0.02)
    _, rates_refresh = sim0.state(REFRESH_T, trophies, r0)
    drops = [rates_refresh[i] / r0[i] for i in range(len(r0)) if r0[i] < 99.99]
    calib = (f"first-refresh rate ratio vs launch: min {min(drops):.3f}, "
             f"max {max(drops):.3f} (all < 1: {all(d < 1 for d in drops)})")

    rows = []
    pooled_v1, pooled_v2 = [], []
    cov_v1, cov_v2, miss_v1, miss_v2 = [], [], [], []
    wins = 0
    for name, sim in scenarios:
        case, answers = sim.build_case(trophies, r0)
        f1 = v1.forecast(case)
        f2 = v2.forecast(case)
        s1 = score(f1["trophies"], answers)
        s2 = score(f2["trophies"], answers)
        if s2["loss"] < s1["loss"]:
            wins += 1
        rows.append((name, s1, s2))
        n_items = len(answers) * 5
        pooled_v1 += [s1["loss"]] * n_items
        pooled_v2 += [s2["loss"]] * n_items
        cov_v1.append(s1["coverage"] * len(answers))
        cov_v2.append(s2["coverage"] * len(answers))
        miss_v1.append(s1["typical_miss"] * len(answers))
        miss_v2.append(s2["typical_miss"] * len(answers))

    n_troph = len(trophies) * len(scenarios)
    pooled = {
        "v1_loss": sum(pooled_v1) / len(pooled_v1),
        "v2_loss": sum(pooled_v2) / len(pooled_v2),
        "v1_coverage": sum(cov_v1) / n_troph,
        "v2_coverage": sum(cov_v2) / n_troph,
        "v1_miss": sum(miss_v1) / n_troph,
        "v2_miss": sum(miss_v2) / n_troph,
        "wins_v2": wins,
        "n_scenarios": len(scenarios),
    }

    # Practice cases with dilution-estimated answers (no reference answers
    # in the kit): answer rate = last * (1 + f*(6.06-1))/6.06.
    practice = {}
    for cid in ("p00", "p01"):
        with open(os.path.join(KIT, "input", f"{cid}.in"), encoding="utf-8") as f:
            case = json.load(f)
        last = case["history"][-1]["rates"]
        for f in (0.10, 0.30):
            factor = (1.0 + f * 5.06) / 6.06
            answers = {}
            for i, tr in enumerate(case["trophies"]):
                r = last[i]
                answers[tr["id"]] = 100.0 if r >= 99.99 else r * factor
            practice[f"{cid}/f={f}"] = {
                "v1": score(v1.forecast(case)["trophies"], answers),
                "v2": score(v2.forecast(case)["trophies"], answers),
            }

    # ---- Report ----
    lines = []
    lines.append("# Backtest v1 vs v2 — cohort simulator (12 scenarios)")
    lines.append("")
    lines.append("Run in this session (fresh code, fresh simulator). "
                 "Simulator: cohort model calibrated on the one recorded "
                 "refresh (638 -> 3,868 players at 04:30 launch night); "
                 "history 2.5 days, horizon 4.5 days (6,510 min).")
    lines.append("")
    lines.append("## Calibration check")
    lines.append("")
    lines.append(calib)
    lines.append("")
    lines.append("## Per-scenario pinball loss (logit scale)")
    lines.append("")
    lines.append("| scenario | v1 loss | v2 loss | v1 cov | v2 cov | v1 miss | v2 miss |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for name, s1, s2 in rows:
        lines.append(f"| {name} | {s1['loss']:.4f} | {s2['loss']:.4f} | "
                     f"{s1['coverage']:.3f} | {s2['coverage']:.3f} | "
                     f"{s1['typical_miss']:.3f} | {s2['typical_miss']:.3f} |")
    lines.append("")
    lines.append("## Pooled (all 12 scenarios, 55 trophies x 5 quantiles each)")
    lines.append("")
    lines.append(f"- v1 loss pooled: **{pooled['v1_loss']:.4f}**")
    lines.append(f"- v2 loss pooled: **{pooled['v2_loss']:.4f}** "
                 f"({100.0 * (pooled['v2_loss'] / pooled['v1_loss'] - 1.0):+.1f}% vs v1)")
    lines.append(f"- v2 wins **{pooled['wins_v2']}/{pooled['n_scenarios']}** scenarios")
    lines.append(f"- coverage 90%: v1 {pooled['v1_coverage']:.3f}, "
                 f"v2 {pooled['v2_coverage']:.3f}")
    lines.append(f"- typical miss (points): v1 {pooled['v1_miss']:.3f}, "
                 f"v2 {pooled['v2_miss']:.3f}")
    lines.append("")
    lines.append("## Practice cases (answers estimated by dilution model)")
    lines.append("")
    lines.append("No reference answers in the kit; answer estimated as "
                 "last rate x (1 + f*(6.06-1))/6.06. v2 must equal v1 here "
                 "(flat regime unchanged).")
    lines.append("")
    for k, v in practice.items():
        lines.append(f"- {k}: v1 loss {v['v1']['loss']:.4f}, "
                     f"v2 loss {v['v2']['loss']:.4f} "
                     f"(identical: {v['v1'] == v['v2']})")
    lines.append("")
    lines.append("## Assumptions recorded (v1 reconstruction)")
    lines.append("")
    lines.append("- v1 jump-refresh variance = 1.0 logit^2 (one refresh-sized "
                 "shock); v1 spread unclipped; v1 100%-trophy spread 1.0.")
    lines.append("- v2 spread = 0.8 * sqrt(resid*leverage + 0.25*days), "
                 "clipped to [0.35, 2.2]; jump term removed (scale 0).")
    lines.append("- v2 structural tracked count: mean of three log-space "
                 "predictions (power-law, exponential on last 24 h, "
                 "saturating-exponential by grid search on k).")
    lines.append("")

    text = "\n".join(lines)
    with open(os.path.join(HERE, "results.md"), "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    return pooled


if __name__ == "__main__":
    run()
