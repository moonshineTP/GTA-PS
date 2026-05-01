@echo off
setlocal enabledelayedexpansion

:: Capture all user arguments once at the start
set "USER_ARGS=%*"

:: Set PYTHONPATH for executables that need it
set "CODE_ROOT=%~dp0.."
set "PYTHONPATH=%CODE_ROOT%\src;%CODE_ROOT%\threads;%CODE_ROOT%\misc"

:: Find and validate artifacts directory
set "ARTIFACTS_DIR="

:: Strategy 1: Check if ARTIFACTS_DIR_ENV is set as environment variable
if defined ARTIFACTS_DIR_ENV (
    if exist "!ARTIFACTS_DIR_ENV!" (
        set "ARTIFACTS_DIR=!ARTIFACTS_DIR_ENV!"
    )
)

:: Strategy 2: Check relative to script location (code\compiled\artifacts)
if not defined ARTIFACTS_DIR (
    if exist "%~dp0artifacts" (
        set "ARTIFACTS_DIR=%~dp0artifacts"
    )
)

:: Strategy 3: Check in code directory (code\artifacts)
if not defined ARTIFACTS_DIR (
    if exist "%~dp0..\artifacts" (
        set "ARTIFACTS_DIR=%~dp0..\artifacts"
    )
)

:: Strategy 4: Check in parent workspace (parent\artifacts)
if not defined ARTIFACTS_DIR (
    if exist "%~dp0..\..\artifacts" (
        set "ARTIFACTS_DIR=%~dp0..\..\artifacts"
    )
)

:: Strategy 5: Check further up the tree
if not defined ARTIFACTS_DIR (
    if exist "%~dp0..\..\..\artifacts" (
        set "ARTIFACTS_DIR=%~dp0..\..\..\artifacts"
    )
)

if not defined ARTIFACTS_DIR (
    echo ERROR: Cannot find artifacts directory
    echo.
    echo Searched in:
    echo   - %~dp0artifacts
    echo   - %~dp0..\artifacts
    echo   - %~dp0..\..\artifacts
    echo   - %~dp0..\..\..\artifacts
    echo.
    echo Solution: Set ARTIFACTS_DIR_ENV environment variable or ensure artifacts folder exists
    echo   Example: set ARTIFACTS_DIR_ENV=D:\workspace\artifacts
    exit /b 1
)

echo ======================================================
echo GTA-PS Compiled Ablations Suite
echo ======================================================
echo Using artifacts from: !ARTIFACTS_DIR!
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
set "DIST_NAME=%EXE_NAME:.exe=.dist%"

echo [%LABEL%]
if exist "!ARTIFACTS_DIR!\!DIST_NAME!\!EXE_NAME!" (
    "!ARTIFACTS_DIR!\!DIST_NAME!\!EXE_NAME!" !USER_ARGS!
) else if exist "!ARTIFACTS_DIR!\!DIST_NAME!\!EXE_NAME:exe=!" (
    "!ARTIFACTS_DIR!\!DIST_NAME!\!EXE_NAME:exe=!" !USER_ARGS!
) else (
    echo ERROR: !EXE_NAME! not found
    echo Expected: !ARTIFACTS_DIR!\!DIST_NAME!\!EXE_NAME! or !ARTIFACTS_DIR!\!DIST_NAME!\!EXE_NAME:exe=!
    exit /b 1
)
echo.
exit /b 0
