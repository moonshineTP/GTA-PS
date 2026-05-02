@echo off
setlocal enabledelayedexpansion

set "USER_ARGS=%*"
set "CODE_ROOT=%~dp0.."
set "PYTHONPATH=%CODE_ROOT%\src;%CODE_ROOT%\threads;%CODE_ROOT%\misc"
set "ARTIFACTS_DIR=%~dp0artifacts"

if not exist "!ARTIFACTS_DIR!" (
    echo ERROR: artifacts directory not found at !ARTIFACTS_DIR!
    exit /b 1
)

echo ======================================================
echo GTA-PS Compiled Ablations Suite
echo ======================================================
echo GTA-PS Compiled Ablations Suite
echo ======================================================

:: Parse --start-from argument
set "START_FROM=1"
set "NEW_USER_ARGS="
for %%a in (!USER_ARGS!) do (
    echo %%a | findstr /r "^--start-from$" >nul 2>&1
    if !errorlevel!==0 (
        set "NEXT_IS_START=1"
    ) else (
        if defined NEXT_IS_START (
            set "START_FROM=%%a"
            set "NEXT_IS_START="
        ) else (
            set "NEW_USER_ARGS=!NEW_USER_ARGS! %%a"
        )
    )
)
set "USER_ARGS=!NEW_USER_ARGS!"
set "INDEX=0"

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
set /a "INDEX+=1"

if !INDEX! LSS !START_FROM! (
    echo [!LABEL!] - skipped ^(start-from=!START_FROM!^)
    exit /b 0
)

echo [!INDEX!/11] !LABEL!

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

echo [!LABEL!]
"!ARTIFACTS_DIR!\!DIST_NAME!\!EXE_NAME!" !USER_ARGS!
echo.
exit /b 0
