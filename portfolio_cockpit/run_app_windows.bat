@echo off
setlocal

cd /d "%~dp0"
set "PYTHON=.venv\Scripts\python.exe"

echo Portfolio Management Cockpit Windows launcher
echo Working directory: %CD%

if not exist "%PYTHON%" (
    echo Local virtual environment not found. Creating .venv...
    where py >nul 2>nul
    if %ERRORLEVEL% EQU 0 (
        echo Trying: py -m venv .venv
        py -m venv .venv
    ) else (
        echo Python launcher py is unavailable.
    )

    if not exist "%PYTHON%" (
        echo Trying: python -m venv .venv
        python -m venv .venv
    )
)

if not exist "%PYTHON%" (
    echo Could not create .venv\Scripts\python.exe. Install Python and retry.
    exit /b 1
)

echo Using Python: %PYTHON%
echo Upgrading pip...
"%PYTHON%" -m pip install --upgrade pip
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%

echo Installing requirements...
"%PYTHON%" -m pip install -r requirements.txt
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%

echo Starting Streamlit with python -m streamlit...
"%PYTHON%" -m streamlit run app.py
