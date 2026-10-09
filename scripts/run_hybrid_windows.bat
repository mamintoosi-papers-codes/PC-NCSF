@echo off
REM Optional Windows starter for the SCOP hybrid clustering stage (report section 8).
REM picks the first python.exe found, in this order:
REM   1. HYBRID_PYTHON environment variable (full path to python.exe)
REM   2. D:\ProgramData\anaconda3\envs\pth\python.exe     (envs the user mentioned)
REM   3. D:\ProgramData\anaconda3\envs\pytorch\python.exe
REM   4. C:\programs\anaconda3\envs\pth\python.exe        (used in the audit report)
REM   5. D:\ProgramData\anaconda3\python.exe              (base env)
REM Every env it probes is smoke-tested first; the first one that passes is used.
setlocal enabledelayedexpansion

cd /d "%~dp0.."

set "CANDIDATES="
if defined HYBRID_PYTHON set "CANDIDATES=!CANDIDATES!;%HYBRID_PYTHON%"
set "CANDIDATES=!CANDIDATES!;D:\ProgramData\anaconda3\envs\pth\python.exe"
set "CANDIDATES=!CANDIDATES!;D:\ProgramData\anaconda3\envs\pytorch\python.exe"
set "CANDIDATES=!CANDIDATES!;C:\programs\anaconda3\envs\pth\python.exe"
set "CANDIDATES=!CANDIDATES!;D:\ProgramData\anaconda3\python.exe"

set "PY="
for %%P in (%CANDIDATES%) do (
    if not defined PY (
        if exist "%%~P" (
            echo Trying env: %%~P
            "%%~P" scripts\check_hybrid_env.py --grid 100
            if !errorlevel! equ 0 (
                set "PY=%%~P"
            ) else (
                echo   ^>smoke test failed, trying next env...
            )
        )
    )
)

if not defined PY (
    echo.
    echo ERROR: no candidate python env passed the smoke test.
    echo Set HYBRID_PYTHON to a python.exe with numpy pandas scipy scikit-learn
    echo ^(and torch+zuko only if a density cache is missing^) and rerun:
    echo     set HYBRID_PYTHON=D:\path\to\envs\your_env\python.exe ^& %~nx0
    exit /b 1
)

echo.
echo Using: %PY%
echo Smoke test passed. Running the full pipeline (results only in results\clustering_hybrid\).
echo.

"%PY%" scripts\clustering_hybrid.py --tier all --grid 100
if errorlevel 1 exit /b 1
"%PY%" scripts\clustering_hybrid.py --tier all --grid 100 --knn
if errorlevel 1 exit /b 1
"%PY%" scripts\clustering_hybrid.py --tier all --grid 100 --strict-run-bound
if errorlevel 1 exit /b 1
"%PY%" scripts\clustering_hybrid.py --tier all --grid 100 --nested
if errorlevel 1 exit /b 1

echo.
echo Done. Outputs are in results\clustering_hybrid\
dir /b results\clustering_hybrid
endlocal
