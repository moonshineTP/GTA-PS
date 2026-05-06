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
echo GTA-PS Compiled RQs Suite
echo ======================================================
echo GTA-PS Compiled RQs Suite
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

call :RunRQ "RQ1 Convergence" "rq1_convergence.exe"
call :RunRQ "RQ1 DP Timeline" "rq1_dp_timeline.exe"
call :RunRQ "RQ1 Dyna Timeline" "rq1_dyna_timeline.exe"
call :RunRQ "RQ2 Allocation" "rq2_allocation.exe"
call :RunRQ "RQ2 Beta" "rq2_beta.exe"
call :RunRQ "RQ3 Tradeoff" "rq3_tradeoff.exe"
call :RunRQ "RQ4 Asymmetry" "rq4_asymmetry.exe"
call :RunRQ "RQ5 Bottlenecks" "rq5_bottlenecks.exe"
call :RunRQ "RQ6 Curl Divergence"

echo Done.
endlocal
exit /b 0

:RunRQ
set "LABEL=%~1"
set "EXE_NAME=%~2"
set /a "INDEX+=1"

if !INDEX! LSS !START_FROM! (
    echo [!LABEL!] - skipped ^(start-from=!START_FROM!^)
    exit /b 0
)

echo [!INDEX!/9] !LABEL!

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
