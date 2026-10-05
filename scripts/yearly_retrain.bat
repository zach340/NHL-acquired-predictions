@echo off
rem Yearly refresh + retrain (Task Scheduler, every July). Output is appended to yearly_retrain.log.
cd /d "%~dp0.."
set PYTHONIOENCODING=utf-8
echo. >> yearly_retrain.log
echo ===== %DATE% %TIME% ===== >> yearly_retrain.log
"C:\Users\Zachc\AppData\Local\Microsoft\WindowsApps\python.exe" scripts\yearly_retrain.py >> yearly_retrain.log 2>&1
