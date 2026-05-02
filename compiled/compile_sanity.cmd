@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
set "CODE_ROOT=%SCRIPT_DIR%.."
set "ARTIFACTS_DIR=%SCRIPT_DIR%artifacts"
set "PYTHON=%CODE_ROOT%\.venv\Scripts\python.exe"
set "PYTHONPATH=%CODE_ROOT%\src;%CODE_ROOT%\threads;%CODE_ROOT%\misc"

echo Starting Sanity Compilation (Windows)...
echo.

set "SANITY=misc\time_benchmark.py misc\convergence_benchmark.py misc\showcase.py"

for %%f in (%SANITY%) do (
    echo --- Compiling %%f ---
    "%PYTHON%" -m nuitka --standalone --output-dir="%ARTIFACTS_DIR%" --assume-yes-for-downloads --jobs=5 --include-package=gta_ps "%CODE_ROOT%\%%f"
)

echo.
echo Sanity benchmarks compiled to %ARTIFACTS_DIR%
endlocal
