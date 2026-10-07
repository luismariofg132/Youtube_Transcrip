@echo off
setlocal
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "TRANSCRIBER_PYTHON="
if exist "%~dp0.venv\Scripts\python.exe" goto virtualenv
if exist "%~dp0python_local.txt" set /p "TRANSCRIBER_PYTHON="<"%~dp0python_local.txt"
if not defined TRANSCRIBER_PYTHON goto bundled
if exist "%~dp0%TRANSCRIBER_PYTHON%" set "TRANSCRIBER_PYTHON=%~dp0%TRANSCRIBER_PYTHON%"
if not exist "%TRANSCRIBER_PYTHON%" goto bundled
"%TRANSCRIBER_PYTHON%" %*
exit /b %errorlevel%
:virtualenv
"%~dp0.venv\Scripts\python.exe" %*
exit /b %errorlevel%
:bundled
if not exist "%~dp0python\python.exe" goto system
"%~dp0python\python.exe" %*
exit /b %errorlevel%
:system
where py >nul 2>nul
if errorlevel 1 goto python
py -3.12 %*
exit /b %errorlevel%
:python
python %*
exit /b %errorlevel%
