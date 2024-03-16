chcp 65001 > NUL
@echo off
setlocal

set PS_CMD=PowerShell -NoProfile -NoLogo -ExecutionPolicy Bypass

set DL_URL=https://github.com/git-for-windows/git/releases/download/v2.44.0.windows.1/MinGit-2.44.0-64-bit.zip
set DL_DST=MinGit-2.44.0-64-bit.zip

set REPO_URL=https://github.com/litagin02/Style-Bert-VITS2

@REM カレントディレクトリをbatファイルのディレクトリに変更
pushd %~dp0

@REM lib フォルダがなければ作成
if not exist lib\ ( mkdir lib )

echo --------------------------------------------------
echo Downloading MinGit...
echo --------------------------------------------------
curl -L %DL_URL% -o "%DL_DST%"
if %errorlevel% neq 0 ( pause & popd & exit /b %errorlevel% )

@REM lib\MinGitフォルダに解凍
echo --------------------------------------------------
echo Extracting MinGit...
echo --------------------------------------------------
%PS_CMD% "Expand-Archive -LiteralPath %DL_DST% -DestinationPath .\lib\MinGit -Force"
if %errorlevel% neq 0 ( pause & popd & exit /b %errorlevel% )

del %DL_DST%

@REM Gitコマンドのパスを設定
set PATH=%~dp0lib\MinGit\cmd;%PATH%

echo --------------------------------------------------
echo Checking Git Installation...
echo --------------------------------------------------
git --version
if %errorlevel% neq 0 ( pause & popd & exit /b %errorlevel% )

echo --------------------------------------------------
echo Cloning repository...
echo --------------------------------------------------
git clone %REPO_URL%
if %errorlevel% neq 0 ( pause & popd & exit /b %errorlevel% )

echo --------------------------------------------------
echo Setting up Python environment...
echo --------------------------------------------------

call Style-Bert-VITS2\scripts\Setup-Python.bat ..\..\lib\python ..\venv
if %errorlevel% neq 0 ( popd & exit /b %errorlevel% )

echo --------------------------------------------------
echo Installing PyTorch...
echo --------------------------------------------------
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
if %errorlevel% neq 0 ( pause & popd & exit /b %errorlevel% )

echo --------------------------------------------------
echo Installing other dependencies...
echo --------------------------------------------------
pip install -r Style-Bert-VITS2\requirements.txt
if %errorlevel% neq 0 ( pause & popd & exit /b %errorlevel% )

echo ----------------------------------------
echo Environment setup is complete. Start downloading the model.
echo ----------------------------------------

@REM Style-Bert-VITS2フォルダに移動
pushd Style-Bert-VITS2

@REM 初期化（必要なモデルのダウンロード）
python initialize.py

echo ----------------------------------------
echo Model download is complete. Start Style-Bert-VITS2 Editor.
echo ----------------------------------------

@REM エディターの起動
python server_editor.py --inbrowser
pause

popd

popd

endlocal
