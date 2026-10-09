@echo off
setlocal
cd /d "%~dp0\.."
echo Running corrected PC-NCSF SCOP clustering from: %CD%
python --version
if errorlevel 1 exit /b 1
python scripts\clustering_hybrid_corrected.py --tier all --grid 100 --eligible-subset --knn
if errorlevel 1 (
  echo ERROR: run failed. See traceback above.
  exit /b 1
)
echo.
echo Results are in results\clustering_hybrid_corrected\
pause
endlocal
