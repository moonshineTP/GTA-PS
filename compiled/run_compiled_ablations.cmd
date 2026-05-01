@echo off
setlocal

:: Capture all user arguments once at the start
set "USER_ARGS=%*"

:: Path to the artifacts directory - works from both code\compiled and code
if exist "%~dp0artifacts" (
    set "ARTIFACTS_DIR=%~dp0artifacts"
) else if exist "%~dp0..\artifacts" (
    set "ARTIFACTS_DIR=%~dp0..\artifacts"
) else (
    set "ARTIFACTS_DIR=artifacts"
)

:: Set PYTHONPATH for executables that need it
set "CODE_ROOT=%~dp0.."
set "PYTHONPATH=%CODE_ROOT%\src;%CODE_ROOT%\threads;%CODE_ROOT%\misc"

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
set "DIST_NAME=%EXE_NAME:.exe=.dist%"

echo [%LABEL%]
if exist "%ARTIFACTS_DIR%\%DIST_NAME%\%EXE_NAME%" (
    "%ARTIFACTS_DIR%\%DIST_NAME%\%EXE_NAME%" %USER_ARGS%
) else if exist "%ARTIFACTS_DIR%\%DIST_NAME%\%EXE_NAME:exe=%" (
    "%ARTIFACTS_DIR%\%DIST_NAME%\%EXE_NAME:exe=%" %USER_ARGS%
) else (
    echo ERROR: %EXE_NAME% not found in %ARTIFACTS_DIR%\%DIST_NAME%
)
echo.
exit /b 0
