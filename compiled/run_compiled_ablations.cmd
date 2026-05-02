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
set "DIST_NAME=!EXE_NAME:.exe=.dist!"

echo [!LABEL!]
"!ARTIFACTS_DIR!\!DIST_NAME!\!EXE_NAME!" !USER_ARGS!
echo.
exit /b 0
