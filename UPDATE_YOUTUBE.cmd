@echo off
if not exist "%~dp0.venv\Scripts\python.exe" goto bundled
"%~dp0.venv\Scripts\python.exe" -m pip install --upgrade "yt-dlp[default]"
goto done
:bundled
call "%~dp0PYTHON_RUNNER.cmd" -m pip install --upgrade --target "%~dp0deps" "yt-dlp[default]"
:done
set "TRANSCRIBER_EXIT_CODE=%errorlevel%"
pause
exit /b %TRANSCRIBER_EXIT_CODE%
