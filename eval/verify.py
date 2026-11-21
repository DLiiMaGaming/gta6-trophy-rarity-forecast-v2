#!/usr/bin/env python3
"""Grader-style verification for rebuild-v2-fresh/main.py.

Checks (run in this session, no network):
  * p00/p01 via subprocess: exit 0, valid shape (55 rows, 5 quantiles,
    non-decreasing, 0..100), <= 65,536 bytes, runtime < 8 s/case,
    byte-identical output across 2 runs;
  * v2 == v1 on practice (flat regime unchanged) via v1_ref.py;
  * edge cases: 1 read, null rates, players_tracked missing, empty history.
"""

import json
import os
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
KIT = os.path.expanduser("~/workspace/makefun/gta6-trophy-rarity-forecast/kit")
MAIN = os.path.join(ROOT, "main.py")
V1 = os.path.join(HERE, "v1_ref.py")


def run_prog(prog, case_path, timeout=30):
    with tempfile.TemporaryDirectory() as td:
        in_d = os.path.join(td, "in")
        out_d = os.path.join(td, "out")
        os.makedirs(in_d)
        os.makedirs(out_d)
        import shutil
        shutil.copy(case_path, in_d)
        env = dict(os.environ, MAKEFUN_INPUT_DIR=in_d, MAKEFUN_OUTPUT_DIR=out_d)
        t0 = time.time()
        p = subprocess.run([sys.executable, prog], env=env,
                           capture_output=True, text=True, timeout=timeout)
        dt = time.time() - t0
        outs = [f for f in os.listdir(out_d) if f.endswith(".out")]
        data = None
        raw = None
        if outs:
            raw = open(os.path.join(out_d, outs[0]), "rb").read()
            data = json.loads(raw)
        return p.returncode, p.stderr.strip(), dt, raw, data


def check_shape(data, n_trophies=55):
    assert isinstance(data, dict) and set(data.keys()) == {"trophies"}, "top keys"
    rows = data["trophies"]
    assert len(rows) == n_trophies, f"rows {len(rows)}"
    ids = sorted(r["id"] for r in rows)
    assert ids == list(range(1, n_trophies + 1)), "ids 1..55"
    for r in rows:
        assert set(r.keys()) == {"id", "q"}, "row keys"
        q = r["q"]
        assert len(q) == 5, "5 quantiles"
        assert all(isinstance(x, (int, float)) and 0 <= x <= 100 for x in q), "range"
        assert all(q[k] <= q[k + 1] for k in range(4)), "non-decreasing"
    return True


def main():
    results = []

    def check(name, ok, detail=""):
        results.append((name, ok, detail))
        print(("PASS" if ok else "FAIL"), "-", name, detail)

    # --- Practice cases, main.py ---
    for cid in ("p00", "p01"):
        case_path = os.path.join(KIT, "input", f"{cid}.in")
        rc1, err1, dt1, raw1, d1 = run_prog(MAIN, case_path)
        rc2, err2, dt2, raw2, d2 = run_prog(MAIN, case_path)
        check(f"{cid} exit 0", rc1 == 0, err1[:200])
        check(f"{cid} runtime < 8s", dt1 < 8, f"{dt1:.2f}s")
        ok_shape = False
        try:
            ok_shape = check_shape(d1) and len(raw1) <= 65536
        except AssertionError as e:
            ok_shape = False
        check(f"{cid} shape valid, <=65536B", ok_shape,
              f"{len(raw1)}B" if raw1 else "no output")
        check(f"{cid} byte-identical x2", raw1 == raw2 and raw1 is not None)

        # v1 comparison on the same case (must be identical: flat regime).
        rcv, errv, _, _, dv = run_prog(V1, case_path)
        check(f"{cid} v2 == v1 (flat regime)", rcv == 0 and d1 == dv,
              errv[:200])

    # --- Edge cases (synthetic, built from p00's trophies) ---
    with open(os.path.join(KIT, "input", "p00.in"), encoding="utf-8") as f:
        base = json.load(f)
    trophies = base["trophies"]

    def synth_case(name, history):
        case = dict(base)
        case["id"] = name
        case["history"] = history
        path = os.path.join(tempfile.mkdtemp(), f"{name}.in")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(case, f)
        return path

    h0 = base["history"][0]
    edge_cases = {
        "edge-1read": [dict(h0)],
        "edge-null-rates": [dict(h0, rates=[None if i % 3 == 0 else r
                                           for i, r in enumerate(h0["rates"])])],
        "edge-no-players": [dict(h0, players_tracked=None)],
        "edge-empty-history": [],
    }
    for name, hist in edge_cases.items():
        path = synth_case(name, hist)
        rc, err, dt, raw, d = run_prog(MAIN, path)
        ok = rc == 0
        try:
            ok = ok and check_shape(d)
        except AssertionError:
            ok = False
        check(f"{name} valid output", ok, f"rc={rc} {dt:.2f}s {err[:150]}")

    # Trend-regime smoke test: 2 distinct snapshots (rates moved).
    h1 = dict(h0)
    h2 = dict(h0, slot="2026-11-19T01:00:00Z", at="2026-11-19T01:00:00.5Z",
              players_tracked=1200,
              rates=[round(r * 0.7, 2) if r < 99.99 else r for r in h0["rates"]])
    path = synth_case("edge-2snap", [h1, h2])
    rc, err, dt, raw, d = run_prog(MAIN, path)
    ok = rc == 0
    try:
        ok = ok and check_shape(d)
    except AssertionError:
        ok = False
    check("edge-2snap (trend regime) valid output", ok,
          f"rc={rc} {dt:.2f}s {err[:150]}")

    failed = [r for r in results if not r[1]]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
