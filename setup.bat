@echo off
setlocal
cd /d "%~dp0"
py -3 --version >nul 2>&1
if errorlevel 1 (
  echo Install Python 3.12 or newer from python.org, including the Python launcher.
  pause
  exit /b 1
)
if not exist ".venv\Scripts\python.exe" py -3 -m venv .venv
if errorlevel 1 exit /b 1
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
  echo Dependency installation failed. See the message above.
  pause
  exit /b 1
)
if not exist ".env" copy ".env.example" ".env" >nul
where ffmpeg >nul 2>&1
if errorlevel 1 echo FFmpeg is missing. See README.md before processing video.
where ffprobe >nul 2>&1
if errorlevel 1 echo FFprobe is missing. See README.md before processing video.
echo Setup finished. Fill in .env locally, then run start_app.bat.
pause
