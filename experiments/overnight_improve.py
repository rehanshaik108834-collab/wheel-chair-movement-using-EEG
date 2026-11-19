"""Overnight orchestrator for Part 1 (experiments/run_improve.py), with the decisions fixed in advance.

1. Screening: every single idea vs BASE on session-T CV (seed 0), 3 GPU lanes in parallel.
2. Keep the ideas with >= +0.5 points over BASE on T-CV (run_improve.KEEP_RULE).
3. If 2+ ideas are kept: their combination on T-CV (seed 0).
4. Final variant = the highest T-CV accuracy among BASE, the kept ideas and the combination.
5. Train on all of session T, test on session E once: 5 seeds (2 lanes), then the summary.
Watchdog: a failed step is retried up to 3 times (finished runs are never redone); a step without
output for 60 minutes is killed and retried. Keeps Windows awake. Status: results/overnight_status_improve.txt
"""
import ctypes
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
SMOKE = "--smoke" in sys.argv
OUT = ROOT / "results" / ("improve_smoke" if SMOKE else "improve")
STATUS = (OUT / "overnight_status.txt") if SMOKE else ROOT / "results" / "overnight_status_improve.txt"
LANES = [["BASE", "FTDB", "AUGNS"], ["AUGSR", "RECIPE"], ["PREPD", "CLEAN"]]
SEEDS_FINAL = [0, 1, 2, 3, 4]
MAX_TRIES, STALL_MINUTES = 3, 60
LOCK = threading.Lock()


def status(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    with LOCK:
        STATUS.parent.mkdir(parents=True, exist_ok=True)
        with open(STATUS, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    print(line, flush=True)


def run(lane, args, threads=3):
    name = " ".join(args)
    out = OUT / f"{lane}.out"
    env = dict(os.environ, OPENBLAS_NUM_THREADS=str(threads), OMP_NUM_THREADS=str(threads), PYTHONUNBUFFERED="1")
    cmd = [PY, "-u", str(ROOT / "experiments" / "run_improve.py"), *args] + (["--smoke"] if SMOKE else [])
    for attempt in range(1, MAX_TRIES + 1):
        status(f"{lane}: START {name} (attempt {attempt}/{MAX_TRIES})")
        OUT.mkdir(parents=True, exist_ok=True)
        with open(out, "a", encoding="utf-8") as f:
            p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT), env=env)
        last, changed = -1, time.time()
        while p.poll() is None:
            time.sleep(20)
            size = out.stat().st_size
            if size != last:
                last, changed = size, time.time()
            elif time.time() - changed > STALL_MINUTES * 60:
                status(f"{lane}: WATCHDOG no output for {STALL_MINUTES} min -> killing and retrying")
                p.kill(); p.wait()
                break
        if p.returncode == 0:
            status(f"{lane}: DONE  {name}")
            return True
        status(f"{lane}: FAILED {name} (exit {p.returncode}); see results/{OUT.name}/{lane}.out")
    return False


def parallel(jobs):
    ok = {}
    ts = [threading.Thread(target=lambda n=n, a=a: ok.__setitem__(n, run(n, a))) for n, a in jobs]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    return all(ok.values())


def main():
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    except Exception:
        pass
    status(f"PART 1 RUN STARTED (pid {os.getpid()}){' [SMOKE]' if SMOKE else ''}")
    if not parallel([(f"lane{i + 1}", ["cv", "--variants", *vs, "--seeds", "0"]) for i, vs in enumerate(LANES)]):
        status("STOPPED: screening failed"); return
    if not run("decide", ["summary"]):
        status("STOPPED: summary failed"); return
    rows = json.loads((OUT / "cv_summary.json").read_text())
    kept = sorted((v for v, r in rows.items() if r["keep"] and "+" not in v), key=lambda v: -rows[v]["cv_acc"])
    status(f"DECISION kept ideas (>= +0.5 on T-CV): {kept or 'none'}")
    if len(kept) >= 2:
        combo = "+".join(kept)
        if not (run("lane1", ["cv", "--variants", combo, "--seeds", "0"]) and run("decide", ["summary"])):
            status("STOPPED: combination failed"); return
        rows = json.loads((OUT / "cv_summary.json").read_text())
    final = max(["BASE"] + kept + (["+".join(kept)] if len(kept) >= 2 else []), key=lambda v: rows[v]["cv_acc"])
    status(f"DECISION final variant: {final} (T-CV {100 * rows[final]['cv_acc']:.2f}% at epoch "
           f"{rows[final]['best_epoch']}; BASE {100 * rows['BASE']['cv_acc']:.2f}%)")
    (OUT / "decision.json").write_text(json.dumps({"kept": kept, "final": final, "epochs": rows[final]["best_epoch"]}, indent=1))
    half = [SEEDS_FINAL[0::2], SEEDS_FINAL[1::2]]
    if not parallel([(f"final{i + 1}", ["t2e", "--variants", final, "--seeds", *map(str, s)]) for i, s in enumerate(half)]):
        status("STOPPED: final T->E failed"); return
    run("decide", ["t2e_summary"])
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
    except Exception:
        pass
    status(f"ALL DONE - results in results/{OUT.name}/cv_summary.md and t2e_summary.md")


if __name__ == "__main__":
    main()
