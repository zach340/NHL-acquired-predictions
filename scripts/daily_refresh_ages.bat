@echo off
cd /d "%~dp0.."
"C:\Users\Zachc\AppData\Local\Microsoft\WindowsApps\python.exe" pipeline\fetch_player_ages.py >> daily_refresh.log 2>&1
