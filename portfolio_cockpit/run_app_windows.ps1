$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

$PythonPath = Join-Path $ScriptDir ".venv\Scripts\python.exe"

Write-Host "Portfolio Management Cockpit Windows launcher"
Write-Host "Working directory: $ScriptDir"

if (-not (Test-Path $PythonPath)) {
    Write-Host "Local virtual environment not found. Creating .venv..."

    if (Get-Command py -ErrorAction SilentlyContinue) {
        Write-Host "Trying: py -m venv .venv"
        & py -m venv .venv
        $venvExitCode = $LASTEXITCODE
    } else {
        $venvExitCode = 1
    }

    if ($venvExitCode -ne 0) {
        Write-Host "Python launcher 'py' failed or is unavailable. Trying: python -m venv .venv"
        & python -m venv .venv
        $venvExitCode = $LASTEXITCODE
    }

    if ($venvExitCode -ne 0 -or -not (Test-Path $PythonPath)) {
        throw "Could not create .venv\Scripts\python.exe. Install Python and retry."
    }
}

Write-Host "Using Python: $PythonPath"
Write-Host "Upgrading pip..."
& $PythonPath -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    throw "pip upgrade failed with exit code $LASTEXITCODE."
}

Write-Host "Installing requirements..."
& $PythonPath -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    throw "requirements installation failed with exit code $LASTEXITCODE."
}

Write-Host "Starting Streamlit with python -m streamlit..."
& $PythonPath -m streamlit run app.py
