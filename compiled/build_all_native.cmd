@echo off
setlocal
cd ..
set "PYTHON=.venv\Scripts\python.exe"
set "PYTHONPATH=src;threads;misc"

echo Starting Full GTA-PS Native Compilation (Safe Limit: 4 Threads)...
echo.

:: Benchmarks
echo [1/3] Compiling Benchmarks...
"%PYTHON%" -m nuitka --standalone --output-dir=compiled\artifacts --mingw64 --assume-yes-for-downloads --jobs=4 --include-package=gta_ps misc\time_benchmark.py
"%PYTHON%" -m nuitka --standalone --output-dir=compiled\artifacts --mingw64 --assume-yes-for-downloads --jobs=4 --include-package=gta_ps misc\convergence_benchmark.py

:: RQs
echo [2/3] Compiling Research Questions (RQs)...
for %%f in (experiments\rq*.py) do (
    echo Compiling %%f...
    "%PYTHON%" -m nuitka --standalone --output-dir=compiled\artifacts --mingw64 --assume-yes-for-downloads --jobs=4 --include-package=gta_ps %%f
)

:: Ablations
echo [3/3] Compiling Ablations...
for %%f in (misc\*_ablation.py) do (
    echo Compiling %%f...
    "%PYTHON%" -m nuitka --standalone --output-dir=compiled\artifacts --mingw64 --assume-yes-for-downloads --jobs=4 --include-package=gta_ps %%f
)

echo Done.
