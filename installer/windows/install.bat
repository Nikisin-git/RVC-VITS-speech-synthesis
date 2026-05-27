@echo off
setlocal enabledelayedexpansion
echo == VoiceGen installer (Windows) ==

where conda >nul 2>nul
if errorlevel 1 (
    echo ERROR: conda is required. Install Miniconda from:
    echo   https://docs.conda.io/en/latest/miniconda.html
    exit /b 1
)

pushd %~dp0\..\..

conda env list | findstr /B "voicegen " >nul
if errorlevel 1 (
    conda env create -f environment.yml
) else (
    conda env update -n voicegen -f environment.yml --prune
)

echo.
echo == Installed. Activate with:
echo   conda activate voicegen ^&^& python -m app.main

popd
endlocal
