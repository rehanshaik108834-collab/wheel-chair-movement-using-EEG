"""Overnight orchestrator with watchdog for experiments/run_mentor.py (no supervision needed).

- Runs two lanes in parallel: CPU lane (her sparse classifier) and GPU lane (EEGNet variants),
  then the final analysis.
- Watchdog: a step that fails is retried up to 3 times (finished work is never redone, every
  phase resumes from saved results); a step with no output for 75 minutes is killed and retried.
- Keeps Windows awake while running.
- Writes progress to results/overnight_status.txt (or results/mentor_smoke/ with --smoke).
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
SMOKE = "--smoke" in sys.argv
PLAN = sys.argv[sys.argv.index("--plan") + 1] if "--plan" in sys.argv else "mentor"
PLANS = {
    "mentor": {"script": "run_mentor.py", "out": "mentor", "status": "overnight_status.txt",
               "cpu": [["sparse_cv"], ["sparse_choose"], ["sparse_p0"], ["sparse_t2e"], ["sparse_loso"]],
               "gpu": [["eeg_cv", "--variants", "BASE", "E1", "E2", "E3"], ["eeg_choose"],
                       ["eeg_t2e", "--variants", "E1", "E2", "E3", "COMBO"], ["eeg_loso"]]},
    "ea": {"script": "run_ea.py", "out": "ea", "status": "overnight_status_ea.txt",
           "cpu": [],
           "gpu": [["within_cv"], ["within_choose"], ["within_t2e"], ["loso"]]},
}
P = PLANS[PLAN]
OUTDIR = ROOT / "results" / (P["out"] + ("_smoke" if SMOKE else ""))
STATUS = (OUTDIR / "overnight_status.txt") if SMOKE else ROOT / "results" / P["status"]
MAX_TRIES, STALL_MINUTES = 3, 75
CPU_LANE, GPU_LANE = P["cpu"], P["gpu"]
LOCK = threading.Lock()
lane_ok = {}


def status(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    with LOCK:
        STATUS.parent.mkdir(parents=True, exist_ok=True)
        with open(STATUS, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    print(line, flush=True)


def run_step(lane, args, threads):
    name = " ".join(args)
    out = OUTDIR / f"{lane}.out"
    env = dict(os.environ, OPENBLAS_NUM_THREADS=str(threads), OMP_NUM_THREADS=str(threads),
               PYTHONUNBUFFERED="1")
    cmd = [PY, "-u", str(ROOT / "experiments" / P["script"]), *args] + (["--smoke"] if SMOKE else [])
    for attempt in range(1, MAX_TRIES + 1):
        status(f"{lane}: START {name} (attempt {attempt}/{MAX_TRIES})")
        with open(out, "a", encoding="utf-8") as f:
            p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT), env=env)
        last_size, last_change = -1, time.time()
        while p.poll() is None:
            time.sleep(20)
            size = out.stat().st_size
            if size != last_size:
                last_size, last_change = size, time.time()
            elif time.time() - last_change > STALL_MINUTES * 60:
                status(f"{lane}: WATCHDOG no output for {STALL_MINUTES} min in {name} -> killing and retrying")
                p.kill()
                p.wait()
                break
        if p.returncode == 0:
            status(f"{lane}: DONE  {name}")
            return True
        status(f"{lane}: FAILED {name} (exit code {p.returncode}); see results/{OUTDIR.name}/{lane}.out")
    return False


def lane(name, steps, threads):
    for args in steps:
        if not run_step(name, args, threads):
            status(f"{name}: STOPPED (step '{' '.join(args)}' failed {MAX_TRIES} times)")
            lane_ok[name] = False
            return
    lane_ok[name] = True
    status(f"{name}: ALL STEPS DONE")


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    keep_awake = 0x80000000 | 0x00000001             # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(keep_awake)
    except Exception:
        pass
    status(f"OVERNIGHT RUN STARTED (plan {PLAN}, pid {os.getpid()}){' [SMOKE TEST]' if SMOKE else ''}")
    threads = [threading.Thread(target=lane, args=(name, steps, n))
               for name, steps, n in (("cpu_lane", CPU_LANE, 6), ("gpu_lane", GPU_LANE, 4)) if steps]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    final_ok = run_step("final", ["final"], 4)
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
    except Exception:
        pass
    if all(lane_ok.values()) and final_ok:
        status(f"ALL DONE - results in results/{OUTDIR.name}/summary.md")
    else:
        status(f"FINISHED WITH PROBLEMS - lanes {lane_ok}, final analysis ok: {final_ok}")


if __name__ == "__main__":
    main()
