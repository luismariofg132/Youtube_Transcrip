@echo off
setlocal
set "TRANSCRIBER_REQUIREMENTS=requirements.txt"
if /i "%~1"=="cuda" set "TRANSCRIBER_REQUIREMENTS=requirements-cuda.txt"
if exist "%~dp0.venv\Scripts\python.exe" goto install
call "%~dp0PYTHON_RUNNER.cmd" -m venv "%~dp0.venv"
if errorlevel 1 goto error
:install
"%~dp0.venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto error
"%~dp0.venv\Scripts\python.exe" -m pip install -r "%~dp0%TRANSCRIBER_REQUIREMENTS%"
set "TRANSCRIBER_EXIT_CODE=%errorlevel%"
pause
exit /b %TRANSCRIBER_EXIT_CODE%
:error
echo Installation failed. See README.md. Python 3.12 and internet access are required.
pause
exit /b 1
