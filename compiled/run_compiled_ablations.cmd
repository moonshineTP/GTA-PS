@echo off
setlocal enabledelayedexpansion

:: Capture all user arguments once at the start
set "USER_ARGS=%*"

:: Set PYTHONPATH for executables that need it
set "CODE_ROOT=%~dp0.."
set "PYTHONPATH=%CODE_ROOT%\src;%CODE_ROOT%\threads;%CODE_ROOT%\misc"

:: Find and validate artifacts directory
set "ARTIFACTS_DIR="

:: Strategy 1: Check if ARTIFACTS_DIR is set as environment variable
if defined ARTIFACTS_DIR_ENV (
    set "ARTIFACTS_DIR=!ARTIFACTS_DIR_ENV!"
)

:: Strategy 2: Check relative to script location
if not defined ARTIFACTS_DIR (
    if exist "%~dp0artifacts" (
        set "ARTIFACTS_DIR=%~dp0artifacts"
    )
)

:: Strategy 3: Check in parent directories from current location
if not defined ARTIFACTS_DIR (
    for /d %%D in (artifacts compiled\artifacts ..\artifacts ..\..\artifacts) do (
        if exist "%%~D" (
            set "ARTIFACTS_DIR=%%~D"
            goto :found_artifacts
        )
    )
)

:: Strategy 4: Search from script's parent directory
if not defined ARTIFACTS_DIR (
    if exist "%~dp0..\artifacts" (
        set "ARTIFACTS_DIR=%~dp0..\artifacts"
    )
)

:found_artifacts
if not defined ARTIFACTS_DIR (
    echo ERROR: Cannot find artifacts directory
    echo.
    echo Searched in:
    echo   - Script directory: %~dp0artifacts
    echo   - Current directory: artifacts, compiled\artifacts, ..\artifacts, ..\..\artifacts
    echo   - Parent of script: %~dp0..\artifacts
    echo.
    echo Solution: Set ARTIFACTS_DIR_ENV environment variable to point to the artifacts folder
    echo   Example: set ARTIFACTS_DIR_ENV=D:\path\to\artifacts
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
