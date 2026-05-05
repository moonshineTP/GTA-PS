@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
set "CODE_ROOT=%SCRIPT_DIR%.."
set "ARTIFACTS_DIR=%SCRIPT_DIR%artifacts"
set "PYTHON=%CODE_ROOT%\.venv\Scripts\python.exe"
set "PYTHONPATH=%CODE_ROOT%\src;%CODE_ROOT%\threads;%CODE_ROOT%\misc"

echo Starting RQs Compilation (Windows)...
echo.

set "RQS=experiments\rq1_proactiveness.py experiments\rq1_dp_timeline.py experiments\rq1_dyna_timeline.py experiments\rq2_allocation.py experiments\rq2_beta_magnitude.py experiments\rq3_mixing_tradeoff.py experiments\rq4_asymmetry.py experiments\rq5_bottlenecks.py experiments\rq6_curl_divergence.py"

for %%f in (%RQS%) do (
    echo --- Compiling %%f ---
    "%PYTHON%" -m nuitka --standalone --output-dir="%ARTIFACTS_DIR%" --assume-yes-for-downloads --jobs=5 --include-package=gta_ps "%CODE_ROOT%\%%f"
)

echo.
echo All RQs compiled to %ARTIFACTS_DIR%
endlocal
