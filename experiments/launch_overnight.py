"""Start experiments/overnight.py as an independent background process.

It keeps running after VS Code / Claude is closed (as long as the laptop stays on).
Usage:  <eeg-bci python> experiments/launch_overnight.py [--smoke]
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DETACHED, NEW_GROUP, BREAKAWAY, NO_WINDOW = 0x00000008, 0x00000200, 0x01000000, 0x08000000

args = [sys.executable, "-u", str(ROOT / "experiments" / "overnight.py"), *sys.argv[1:]]
log = open(ROOT / "results" / "overnight_launcher.out", "a")
for flags in (DETACHED | NEW_GROUP | BREAKAWAY | NO_WINDOW, DETACHED | NEW_GROUP | NO_WINDOW):
    try:
        p = subprocess.Popen(args, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL, creationflags=flags, close_fds=True)
        print(f"launched overnight.py, pid {p.pid} (breakaway={'yes' if flags & BREAKAWAY else 'no'})")
        break
    except OSError as e:
        print(f"launch attempt failed ({e}); retrying without job breakaway")
