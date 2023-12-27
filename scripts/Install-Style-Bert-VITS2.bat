chcp 65001 > NUL

@REM https://github.com/Zuntan03/EasyBertVits2 より引用・改変

@echo off
pushd %~dp0
set PS_CMD=PowerShell -Version 5.1 -ExecutionPolicy Bypass

set CURL_CMD=C:\Windows\System32\curl.exe
if not exist %CURL_CMD% (
	echo [ERROR] %CURL_CMD% が見つかりません。
	pause & popd & exit /b 1
)

if not exist lib\ ( mkdir lib )

%CURL_CMD% -Lo Style-Bert-VITS2.zip^
	https://github.com/litagin02/Style-Bert-VITS2/archive/refs/heads/master.zip
if %errorlevel% neq 0 ( pause & popd & exit /b %errorlevel% )

%PS_CMD% Expand-Archive -Path Style-Bert-VITS2.zip -DestinationPath . -Force
if %errorlevel% neq 0 ( pause & popd & exit /b %errorlevel% )

del Style-Bert-VITS2.zip
if %errorlevel% neq 0 ( pause & popd & exit /b %errorlevel% )

ren Style-Bert-VITS2-master Style-Bert-VITS2

call Style-Bert-VITS2\scripts\Setup-Python.bat ..\..\lib\python ..\venv
if %errorlevel% neq 0 ( popd & exit /b %errorlevel% )

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
if %errorlevel% neq 0 ( pause & popd & exit /b %errorlevel% )

pip install -r Style-Bert-VITS2\requirements.txt
if %errorlevel% neq 0 ( pause & popd & exit /b %errorlevel% )

pushd Style-Bert-VITS2
python initialize.py
popd

popd

pause
