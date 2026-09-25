<#
.SYNOPSIS
  ABP launcher for Windows (PowerShell 5.1 or 7+).

.DESCRIPTION
  Runs "python -m abp <arguments>" with UTF-8 console I/O so that Chinese
  characters and spaces in paths and IDs work, streams the output to the
  console, keeps a readable UTF-8 launcher log, and returns ABP's exit status
  unchanged (0 ok, 2 usage, 3 input, 4 QC, 5 model, 6 numerical, 7 resource,
  130 cancelled, 1 internal).

  Log folder: %LOCALAPPDATA%\ABP\logs  (one file per invocation)
  Python:     $env:ABP_PYTHON if set, otherwise "python" on PATH.

.EXAMPLE
  .\abp.ps1 run "D:\育种 数据\analysis.toml" --out "D:\育种 数据\结果"
#>
# No param() block on purpose: a non-advanced script receives every token,
# including GNU-style options such as --out, unchanged in $args.
$AbpArgs = @($args)

[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'

$logDir = Join-Path $env:LOCALAPPDATA 'ABP\logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$log = Join-Path $logDir ("abp-" + (Get-Date -Format 'yyyyMMdd-HHmmss-fff') + ".log")
$python = if ($env:ABP_PYTHON) { $env:ABP_PYTHON } else { 'python' }

$quoted = ($AbpArgs | ForEach-Object { '"' + $_ + '"' }) -join ' '
Add-Content -Path $log -Encoding utf8 -Value ("started: " + (Get-Date -Format o))
Add-Content -Path $log -Encoding utf8 -Value ("command: $python -m abp $quoted")

if (-not (Get-Command $python -ErrorAction SilentlyContinue)) {
    $msg = "error: Python interpreter '$python' not found. Install Python 3.11+ or set ABP_PYTHON."
    Write-Host $msg
    Add-Content -Path $log -Encoding utf8 -Value $msg
    Add-Content -Path $log -Encoding utf8 -Value "exit_code: 9009"
    exit 9009
}

$ErrorActionPreference = 'Continue'   # stderr lines from Python must not abort the pipeline
& $python -m abp @AbpArgs 2>&1 | ForEach-Object {
    $line = $_.ToString()
    Write-Host $line
    Add-Content -Path $log -Encoding utf8 -Value $line
}
$rc = $LASTEXITCODE
Add-Content -Path $log -Encoding utf8 -Value ("exit_code: $rc")
Add-Content -Path $log -Encoding utf8 -Value ("finished: " + (Get-Date -Format o))
if ($rc -ne 0) { Write-Host "ABP exited with status $rc; launcher log: $log" }
exit $rc
