"""Orchestrator for Part 1.6 + Stage B on the improved model (decisions fixed in advance, see run_ens.py).

1. 10 more final models (AUGSR, seeds 5-14, train on all T) -> 15 models   (run_improve.py t2e, 3 lanes)
2. T-CV with held-out logits + MC dropout, seeds 0-4                        (run_ens.py cv, 3 lanes by subject)
3. Export E logits / MC of the 15 models, 4. final summary (rule chosen on T-CV, E once, Stage B).
Finished runs are never redone; a failed step is retried up to 3 times; 60 min without output -> kill + retry.
Status: results/overnight_status_ens.txt
"""
import ctypes
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
OUT = ROOT / "results" / "stageB_aug"
STATUS = ROOT / "results" / "overnight_status_ens.txt"
MAX_TRIES, STALL_MINUTES = 3, 60
LOCK = threading.Lock()


def status(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    with LOCK:
        with open(STATUS, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    print(line, flush=True)


def run(lane, script, args, threads=3):
    name = f"{script} {' '.join(args)}"
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"{lane}.out"
    env = dict(os.environ, OPENBLAS_NUM_THREADS=str(threads), OMP_NUM_THREADS=str(threads), PYTHONUNBUFFERED="1")
    cmd = [PY, "-u", str(ROOT / "experiments" / script), *args]
    for attempt in range(1, MAX_TRIES + 1):
        status(f"{lane}: START {name} (attempt {attempt}/{MAX_TRIES})")
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
        status(f"{lane}: FAILED {name} (exit {p.returncode}); see results/stageB_aug/{lane}.out")
    return False


def parallel(jobs):
    ok = {}
    ts = [threading.Thread(target=lambda n=n, s=s, a=a: ok.__setitem__(n, run(n, s, a))) for n, s, a in jobs]
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
    status(f"PART 1.6 + STAGE B (improved model) STARTED (pid {os.getpid()})")
    seeds = [["5", "8", "11", "14"], ["6", "9", "12"], ["7", "10", "13"]]
    if not parallel([(f"lane{i + 1}", "run_improve.py", ["t2e", "--variants", "AUGSR", "--seeds", *s])
                     for i, s in enumerate(seeds)]):
        status("STOPPED: extra final models failed"); return
    subj = [["A01", "A04", "A07"], ["A02", "A05", "A08"], ["A03", "A06", "A09"]]
    if not parallel([(f"lane{i + 1}", "run_ens.py", ["cv", "--subjects", *s]) for i, s in enumerate(subj)]):
        status("STOPPED: T-CV failed"); return
    if not (run("lane1", "run_ens.py", ["export"]) and run("lane1", "run_ens.py", ["final"])):
        status("STOPPED: export/final failed"); return
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
    except Exception:
        pass
    status("ALL DONE - results in results/stageB_aug/summary.md")


if __name__ == "__main__":
    main()
