"""Orchestrator for B4-B6 (experiments/run_cross.py): train 27 LOSO models on 3 GPU lanes, then adapt + final.
Same watchdog / retry / resume / keep-awake as overnight_ens.py. Status: results/overnight_status_cross.txt"""
import ctypes
import os
from pathlib import Path

import overnight_ens as oe

ROOT = Path(__file__).resolve().parents[1]
oe.OUT = ROOT / "results" / "cross"
oe.STATUS = ROOT / "results" / "overnight_status_cross.txt"


def main():
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    except Exception:
        pass
    oe.status(f"B4-B6 CROSS-SUBJECT STARTED (pid {os.getpid()})")
    subj = [["A01", "A04", "A07"], ["A02", "A05", "A08"], ["A03", "A06", "A09"]]
    if not oe.parallel([(f"lane{i + 1}", "run_cross.py", ["train", "--subjects", *s]) for i, s in enumerate(subj)]):
        oe.status("STOPPED: training failed"); return
    if not (oe.run("lane1", "run_cross.py", ["adapt"]) and oe.run("lane1", "run_cross.py", ["final"])):
        oe.status("STOPPED: adapt/final failed"); return
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
    except Exception:
        pass
    oe.status("ALL DONE - results in results/cross/summary.md")


if __name__ == "__main__":
    main()
