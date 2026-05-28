@echo off
chcp 65001 >nul
title LAMMPS Modeling Toolbox
cd /d "%~dp0"
echo ============================
echo   LAMMPS Modeling Toolbox
echo   Location: liblammps.dll (Python)
echo   Work dir: %~dp0
echo ============================
echo.
echo  1. Run Hydrogel model
echo  2. Run MXene Ti3C2Tx model
echo  3. Open work folder
echo  4. Exit
echo.
set /p choice="Enter (1-4): "

if "%choice%"=="1" (
    echo.
    echo [RUN] Hydrogel modeling...
    python build_hydrogel.py
    echo.
    pause
    goto :eof
)
if "%choice%"=="2" (
    echo.
    echo [RUN] MXene Ti3C2Tx modeling...
    python build_mxene.py
    echo.
    pause
    goto :eof
)
if "%choice%"=="3" (
    explorer "%~dp0"
    goto :eof
)
if "%choice%"=="4" exit /b

echo Invalid input
pause
