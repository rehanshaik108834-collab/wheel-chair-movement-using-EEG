# Runs Phases 2-5 of notebook 04 in order. Safe to re-run: finished runs are skipped.
#   powershell -ExecutionPolicy Bypass -File experiments\run_all.ps1
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$py = "$env:USERPROFILE\anaconda3\envs\eeg-bci\python.exe"

function Step($name, $argList, $out) {
    & $py -W ignore experiments\run_experiments.py @argList *> "results\$out"
    if ($LASTEXITCODE -ne 0) { throw "$name failed (exit $LASTEXITCODE), see results\$out" }
}

Step "Phase 2: select"  @("select")                                         "phase2_select.out"
Step "Phase 2: choose"  @("choose")                                         "phase2_choose.out"
Step "Phase 3: T->E"    @("t2e", "--seeds", "0", "1", "2", "3", "4")        "phase3_t2e.out"
Step "Phase 4: all"     @("t2e", "--configs", "A", "B", "C", "D", "--seeds", "0", "1", "2", "3", "4") "phase4_t2e_all.out"
Step "Phase 5: LOSO"    @("loso", "--seeds", "0", "1", "2")                 "phase5_loso.out"
"ALL PHASES FINISHED"
