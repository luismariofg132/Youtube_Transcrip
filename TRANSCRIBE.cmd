@echo off
call "%~dp0PYTHON_RUNNER.cmd" "%~dp0transcribe.py" %*
set "TRANSCRIBER_EXIT_CODE=%errorlevel%"
pause
exit /b %TRANSCRIBER_EXIT_CODE%
