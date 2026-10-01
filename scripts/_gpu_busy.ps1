# Exit code 0 when a GPU job of this repository is running, 1 when the GPU is free for the next one.
# GPU jobs here are the trainers and the probes that default to CUDA; CPU-only probes (--device cpu) never block.
$busy = Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object {
    $_.CommandLine -match 'mline\.(emergence|decoy_train|rope_probe|channel_probe)' -or
    ($_.CommandLine -match 'mline\.longrun' -and $_.CommandLine -notmatch '--device cpu') -or
    ($_.CommandLine -match 'mline\.(state_swap|decoy_probe)' -and $_.CommandLine -match '--device cuda')
}
if ($busy) { exit 0 } else { exit 1 }
