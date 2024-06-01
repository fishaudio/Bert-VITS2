chcp 65001 > NUL
@echo off

@REM エラーコードを遅延評価するために設定
setlocal enabledelayedexpansion

@REM PowerShellのコマンド
set PS_CMD=PowerShell -Version 5.1 -ExecutionPolicy Bypass

@REM PortableGitのURLと保存先
set DL_URL=https://github.com/git-for-windows/git/releases/download/v2.44.0.windows.1/PortableGit-2.44.0-64-bit.7z.exe
set DL_DST=%~dp0lib\PortableGit-2.44.0-64-bit.7z.exe

@REM Style-Bert-VITS2のリポジトリURL
set REPO_URL=https://github.com/litagin02/Style-Bert-VITS2

@REM カレントディレクトリをbatファイルのディレクトリに変更
pushd %~dp0

@REM lib フォルダがなければ作成
if not exist lib\ ( mkdir lib )

echo --------------------------------------------------
echo PS_CMD: %PS_CMD%
echo DL_URL: %DL_URL%
echo DL_DST: %DL_DST%
echo REPO_URL: %REPO_URL%
echo --------------------------------------------------
echo.
echo --------------------------------------------------
echo Checking Git Installation...
echo --------------------------------------------------
echo Executing: git --version
git --version
if !errorlevel! neq 0 (
	echo --------------------------------------------------
	echo Git is not installed, so download and use PortableGit.
	echo Downloading PortableGit...
	echo --------------------------------------------------
	echo Executing: curl -L %DL_URL% -o "%DL_DST%"
	curl -L %DL_URL% -o "%DL_DST%"
	if !errorlevel! neq 0 ( pause & popd & exit /b !errorlevel! )

	echo --------------------------------------------------
	echo Extracting PortableGit...
	echo --------------------------------------------------
	echo Executing: "%DL_DST%" -y
	"%DL_DST%" -y
	if !errorlevel! neq 0 ( pause & popd & exit /b !errorlevel! )

	echo --------------------------------------------------
	echo Removing %DL_DST%...
	echo --------------------------------------------------
	echo Executing: del "%DL_DST%"
	del "%DL_DST%"
	if !errorlevel! neq 0 ( pause & popd & exit /b !errorlevel! )

	@REM Gitコマンドのパスを設定
	echo --------------------------------------------------
	echo Setting up PATH...
	echo --------------------------------------------------
	echo Executing: set "PATH=%~dp0lib\PortableGit\bin;%PATH%"
	set "PATH=%~dp0lib\PortableGit\bin;%PATH%"
	if !errorlevel! neq 0 ( pause & popd & exit /b !errorlevel! )

	echo --------------------------------------------------
	echo Checking Git Installation...
	echo --------------------------------------------------
	echo Executing: git --version
	git --version
	if !errorlevel! neq 0 ( pause & popd & exit /b !errorlevel! )
)

echo --------------------------------------------------
echo Cloning repository...
echo --------------------------------------------------
echo Executing: git clone %REPO_URL%
git clone %REPO_URL%
if !errorlevel! neq 0 ( pause & popd & exit /b !errorlevel! )

@REM Pythonのセットアップ、仮想環境が有効化されて戻って来る
echo --------------------------------------------------
echo Setting up Python environment...
echo --------------------------------------------------
echo Executing: call Setup-Python.bat ".\lib\python" ".\Style-Bert-VITS2\venv"
call Setup-Python.bat ".\lib\python" ".\Style-Bert-VITS2\venv"
if !errorlevel! neq 0 ( popd & exit /b !errorlevel! )

@REM Style-Bert-VITS2フォルダに移動
pushd Style-Bert-VITS2

@REM 後で消す！！！！！！！！！！
@REM git checkout dev
@REM 後で消す！！！！！！！！！！

echo --------------------------------------------------
echo Activating the virtual environment...
echo --------------------------------------------------
echo Executing: call ".\venv\Scripts\activate.bat"
call ".\venv\Scripts\activate.bat"
if !errorlevel! neq 0 ( popd & exit /b !errorlevel! )

echo --------------------------------------------------
echo Installing package manager uv...
echo --------------------------------------------------
echo Executing: pip install uv
pip install uv
if !errorlevel! neq 0 ( pause & popd & exit /b !errorlevel! )

echo --------------------------------------------------
echo Installing dependencies...
echo --------------------------------------------------
echo Executing: uv pip install -r requirements-infer.txt
uv pip install -r requirements-infer.txt
if !errorlevel! neq 0 ( pause & popd & exit /b !errorlevel! )

echo ----------------------------------------
echo Environment setup is complete. Start downloading the model.
echo ----------------------------------------
echo Executing: python initialize.py
python initialize.py --only_infer

echo ----------------------------------------
echo Model download is complete. Start Style-Bert-VITS2 Editor.
echo ----------------------------------------
echo Executing: python server_editor.py --inbrowser
python server_editor.py --inbrowser
pause

popd

popd

endlocal
