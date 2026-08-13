@echo off
cd /d "%~dp0"
python -m pip install -r requirements.txt
python -m PyInstaller --noconfirm --clean TraceSheetStudio.spec
if errorlevel 1 pause
