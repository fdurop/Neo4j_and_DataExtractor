$ErrorActionPreference = 'Stop'
$appDirectory = $PSScriptRoot
$runtimeTempPath = Join-Path $appDirectory 'runtime\tmp'
New-Item -ItemType Directory -Path $runtimeTempPath -Force | Out-Null
$env:TEMP = $runtimeTempPath
$env:TMP = $runtimeTempPath
python -m venv (Join-Path $appDirectory '.venv')
if ($LASTEXITCODE -ne 0) { throw 'Python environment creation failed.' }
$requirementsPath = Join-Path $appDirectory 'requirements.lock.txt'
if (-not (Test-Path -LiteralPath $requirementsPath)) {
    $requirementsPath = Join-Path $appDirectory 'requirements.txt'
}
& (Join-Path $appDirectory '.venv\Scripts\python.exe') -m pip install -r $requirementsPath
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
