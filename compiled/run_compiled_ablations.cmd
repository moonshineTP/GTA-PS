@echo off
setlocal

:: Capture all user arguments once at the start
set "USER_ARGS=%*"

:: Path to the artifacts directory relative to this script
set "ARTIFACTS_DIR=%~dp0artifacts"

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

echo [%LABEL%]
"%ARTIFACTS_DIR%\%EXE_NAME:.exe=.dist%\%EXE_NAME%" %USER_ARGS%
echo.
exit /b 0
