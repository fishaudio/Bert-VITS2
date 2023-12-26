@REM https://github.com/Zuntan03/EasyBertVits2 より引用・改変

@echo off
chcp 65001 > NUL
pushd %~dp0..
set PS_CMD=PowerShell -Version 5.1 -ExecutionPolicy Bypass

set CURL_CMD=C:\Windows\System32\curl.exe
if not exist %CURL_CMD% (
	echo [ERROR] %CURL_CMD% が見つかりません。
	pause & popd & exit /b 1
)

echo 以下の配布元から関連ファイルをダウンロードして使用します（URL を Ctrl + クリックで開けます）。
echo https://www.python.org/
echo https://github.com/pypa/get-pip
echo https://github.com/litagin02/Style-Bert-VITS2/
echo よろしいですか？ [y/n]
set /p YES_OR_NO=
if /i not "%YES_OR_NO%" == "y" ( popd & exit /b 1 )

if not exist lib\ ( mkdir lib )

if not exist Style-Bert-VITS2\ (
	%CURL_CMD% -Lo Style-Bert-VITS2.zip https://github.com/litagin02/Style-Bert-VITS2/archive/refs/heads/master.zip
	if %errorlevel% neq 0 ( pause & popd & exit /b %errorlevel% )

	%PS_CMD% Expand-Archive -Path Style-Bert-VITS2.zip -DestinationPath . -Force
	if %errorlevel% neq 0 ( pause & popd & exit /b %errorlevel% )

	del Style-Bert-VITS2.zip
	if %errorlevel% neq 0 ( pause & popd & exit /b %errorlevel% )
)

call %~dp0Setup-Python.bat ..\lib\python ..\Bert-VITS2\venv
if %errorlevel% neq 0 ( popd & exit /b %errorlevel% )

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
if %errorlevel% neq 0 ( pause & popd & exit /b %errorlevel% )

pip install -r Bert-VITS2\requirements.txt
if %errorlevel% neq 0 ( pause & popd & exit /b %errorlevel% )

pushd Bert-VITS2
python initialize.py

popd
