# Stop the experiment queues for the user (2026-10-24: "yarin minecraft gunum"): the queue runners first, so none of them logs a failure and
# moves on to its next run, then the experiment processes. Every run checkpoints; scripts/resume_after_user.sh relaunches the queues.
# -DryRun lists what would be stopped and stops nothing.
param([switch]$DryRun)
$runners = @(Get-CimInstance Win32_Process -Filter "Name='bash.exe'" | Where-Object { $_.CommandLine -match 'scripts/run_[A-Za-z0-9_]+\.sh' })
foreach ($r in $runners) {
    "runner   $($r.ProcessId)  $($r.CommandLine.Substring(0, [Math]::Min(120, $r.CommandLine.Length)))"
    if (-not $DryRun) { Stop-Process -Id $r.ProcessId -Force -Confirm:$false -ErrorAction SilentlyContinue }
}
if (-not $DryRun) { Start-Sleep -Seconds 3 }
$runs = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match ' -m (mline|horizon)\.' })
foreach ($p in $runs) {
    "process  $($p.ProcessId)  $($p.CommandLine.Substring(0, [Math]::Min(160, $p.CommandLine.Length)))"
    if (-not $DryRun) { Stop-Process -Id $p.ProcessId -Force -Confirm:$false -ErrorAction SilentlyContinue }
}
"stopped: $($runners.Count) runner(s), $($runs.Count) process(es)" + $(if ($DryRun) { ' (dry run: nothing stopped)' } else { '' })
