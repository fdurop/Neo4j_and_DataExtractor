param([int]$Port = 8769)
$ErrorActionPreference = 'Stop'
$appDirectory = $PSScriptRoot
$pythonPath = Join-Path $appDirectory '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw 'Run setup.ps1 first to create the local Python environment.'
}
& $pythonPath -X utf8 (Join-Path $appDirectory 'serve.py') --port $Port
