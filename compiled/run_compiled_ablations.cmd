@echo off
setlocal enabledelayedexpansion

:: If first argument looks like a path, use it as executables directory
if not "%~1"=="" (
    if exist "%~1" (
        set "EXES_DIR=%~1"
        set "USER_ARGS=%~2 %~3 %~4 %~5 %~6 %~7 %~8 %~9"
        goto :found_exes
    )
)

:: Otherwise capture all args as user arguments
set "USER_ARGS=%*"

:: Set PYTHONPATH for executables that need it
set "CODE_ROOT=%~dp0.."
set "PYTHONPATH=%CODE_ROOT%\src;%CODE_ROOT%\threads;%CODE_ROOT%\misc"

:: Find executables directory
set "EXES_DIR="

:: Strategy 1: Check if EXES_DIR_ENV is set
if defined EXES_DIR_ENV (
    if exist "!EXES_DIR_ENV!" (
        set "EXES_DIR=!EXES_DIR_ENV!"
    )
)

:: Strategy 2: Check for artifacts structure (compiled\artifacts\*.dist\*.exe)
if not defined EXES_DIR (
    if exist "%~dp0artifacts" (
        set "EXES_DIR=%~dp0artifacts"
    )
)

:: Strategy 3: Check parent directories
if not defined EXES_DIR (
    if exist "%~dp0..\artifacts" (
        set "EXES_DIR=%~dp0..\artifacts"
    )
)

if not defined EXES_DIR (
    if exist "%~dp0..\..\artifacts" (
        set "EXES_DIR=%~dp0..\..\artifacts"
    )
)

:found_exes
if not defined EXES_DIR (
    echo ERROR: Cannot find executables
    echo.
    echo Usage:
    echo   run_compiled_ablations.cmd [path_to_exes] [args...]
    echo.
    echo Examples:
    echo   run_compiled_ablations.cmd
    echo   run_compiled_ablations.cmd D:\executables
    echo   run_compiled_ablations.cmd "C:\Program Files\GTA-PS\exes"
    echo.
    echo Or set environment variable:
    echo   set EXES_DIR_ENV=D:\path\to\exes
    exit /b 1
)

echo ======================================================
echo GTA-PS Compiled Ablations Suite
echo ======================================================
echo Using executables from: !EXES_DIR!
echo.
echo GTA-PS Compiled Ablations Suite
echo ======================================================


call :RunAbl "Smoothness Ablation" "smoothness_ablation.exe"
call :RunAbl "Mixing Threshold Ablation" "mixing_threshold_ablation.exe"
call :RunAbl "Shift-rate Ablation" "shift_rate_ablation.exe"
call :RunAbl "Boltzmann-b Ablation" "boltzmann_b_ablation.exe"
call :RunAbl "Scheduler Ablation" "scheduler_ablation.exe"
call :RunAbl "Gamma Ablation" "gamma_ablation.exe"
call :RunAbl "Mixing Ablation" "mixing_ablation.exe"
call :RunAbl "Periodicity Ablation" "periodicity_ablation.exe"
call :RunAbl "Reward-support Ablation" "reward_support_ablation.exe"
call :RunAbl "Cumulative Ablation" "cumulative_ablation.exe"
call :RunAbl "Showcase" "showcase.exe"

echo Done.
endlocal
exit /b 0

:RunAbl
set "LABEL=%~1"
set "EXE_NAME=%~2"

echo [%LABEL%]

:: Try to run executable directly from EXES_DIR
if exist "!EXES_DIR!\!EXE_NAME!" (
    "!EXES_DIR!\!EXE_NAME!" !USER_ARGS!
    exit /b 0
)

:: Try with .exe extension
if exist "!EXES_DIR!\!EXE_NAME!.exe" (
    "!EXES_DIR!\!EXE_NAME!.exe" !USER_ARGS!
    exit /b 0
)

:: Try artifacts structure: EXES_DIR\name.dist\name.exe
set "DIST_NAME=!EXE_NAME:.exe=.dist!"
if exist "!EXES_DIR!\!DIST_NAME!\!EXE_NAME!" (
    "!EXES_DIR!\!DIST_NAME!\!EXE_NAME!" !USER_ARGS!
    exit /b 0
)

:: Try artifacts structure without .exe
if exist "!EXES_DIR!\!DIST_NAME!\!DIST_NAME!" (
    "!EXES_DIR!\!DIST_NAME!\!DIST_NAME!" !USER_ARGS!
    exit /b 0
)

echo ERROR: !EXE_NAME! not found
echo Searched in:
echo   - !EXES_DIR!\!EXE_NAME!
echo   - !EXES_DIR!\!EXE_NAME!.exe
echo   - !EXES_DIR!\!DIST_NAME!\!EXE_NAME!
echo.
exit /b 1
