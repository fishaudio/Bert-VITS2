chcp 65001 > NUL
@REM ↑文字コードのおまじないらしい

@REM https://github.com/Zuntan03/EasyBertVits2 より引用・改変


@REM 2. Style-Bert-VITS2.zip を解凍（名前がBert-VITS2-masterになる）
@REM 3. Style-Bert-VITS2-master を Style-Bert-VITS2 にリネーム


@echo off
pushd %~dp0
set PS_CMD=PowerShell -Version 5.1 -ExecutionPolicy Bypass

set CURL_CMD=C:\Windows\System32\curl.exe
if not exist %CURL_CMD% (
	echo [ERROR] %CURL_CMD% が見つかりません。
	pause & popd & exit /b 1
)

@REM lib フォルダがなければ作成
if not exist lib\ ( mkdir lib )

@REM Style-Bert-VITS2.zip をGitHubのmasterの最新のものをダウンロード
%CURL_CMD% -Lo Style-Bert-VITS2.zip^
	https://github.com/litagin02/Style-Bert-VITS2/archive/refs/heads/master.zip
if %errorlevel% neq 0 ( pause & popd & exit /b %errorlevel% )

@REM Style-Bert-VITS2.zip を解凍（フォルダ名前がBert-VITS2-masterになる）
%PS_CMD% Expand-Archive -Path Style-Bert-VITS2.zip -DestinationPath . -Force
if %errorlevel% neq 0 ( pause & popd & exit /b %errorlevel% )

@REM 元のzipを削除しStyle-Bert-VITS2-master を Style-Bert-VITS2 にリネーム
del Style-Bert-VITS2.zip
if %errorlevel% neq 0 ( pause & popd & exit /b %errorlevel% )

ren Style-Bert-VITS2-master Style-Bert-VITS2

@REM Pythonと仮想環境のセットアップを呼び出す（仮想環境が有効化されて戻ってくる）
call Style-Bert-VITS2\scripts\Setup-Python.bat ..\..\lib\python ..\venv
if %errorlevel% neq 0 ( popd & exit /b %errorlevel% )

@REM 依存関係インストール
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
if %errorlevel% neq 0 ( pause & popd & exit /b %errorlevel% )

pip install -r Style-Bert-VITS2\requirements.txt
if %errorlevel% neq 0 ( pause & popd & exit /b %errorlevel% )

@REM Style-Bert-VITS2フォルダに移動
pushd Style-Bert-VITS2

@REM 初期化（必要なモデルのダウンロード）
python initialize.py

@REM 音声合成WebUIの起動
python app.py

pause

popd

popd

