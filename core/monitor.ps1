param([int]$TrackedProcessId)
$outPath = Join-Path $PSScriptRoot 'resource-samples.jsonl'
while ($true) {
    $trackedProcess = Get-Process -Id $TrackedProcessId -ErrorAction SilentlyContinue
    if ($null -eq $trackedProcess) { break }
    $row = @{ utc = [DateTime]::UtcNow.ToString('o'); process_id = $TrackedProcessId; working_set_bytes = $trackedProcess.WorkingSet64; peak_working_set_bytes = $trackedProcess.PeakWorkingSet64; cpu_seconds = $trackedProcess.TotalProcessorTime.TotalSeconds; scope = 'whole primary Python process, not activation-only memory' }
    $row | ConvertTo-Json -Compress | Add-Content -LiteralPath $outPath -Encoding utf8
    Start-Sleep -Seconds 2
}
