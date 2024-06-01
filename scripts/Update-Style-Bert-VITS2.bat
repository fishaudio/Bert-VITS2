chcp 65001 > NUL
@echo off

@REM エラーコードを遅延評価するために設定
setlocal enabledelayedexpansion

pushd %~dp0


pushd Style-Bert-VITS2

echo --------------------------------------------------
echo Checking Git Installation...
echo --------------------------------------------------
git --version
if !errorlevel! neq 0 (
	echo --------------------------------------------------
	echo Global Git is not installed, so use PortableGit.
	echo Setting up PATH...
	echo --------------------------------------------------
	echo Executing: set "PATH=%~dp0lib\PortableGit\bin;%PATH%"
	set "PATH=%~dp0lib\PortableGit\bin;%PATH%"

	echo --------------------------------------------------
	echo Checking Git Installation...
	echo --------------------------------------------------
	echo Executing: git --version
	git --version
	if !errorlevel! neq 0 ( pause & popd & exit /b !errorlevel! )
)

echo --------------------------------------------------
echo Git pull...
echo --------------------------------------------------
git pull
if !errorlevel! neq 0 ( pause & popd & exit /b !errorlevel! )

@REM 仮想環境のpip requirements.txtを更新

echo --------------------------------------------------
echo Activating virtual environment...
echo --------------------------------------------------
echo Executing: call ".\venv\Scripts\activate.bat"
call ".\venv\Scripts\activate.bat"
if !errorlevel! neq 0 ( pause & popd & exit /b !errorlevel! )

echo --------------------------------------------------
echo Installing uv...
echo --------------------------------------------------
echo Executing: pip install -U uv
pip install -U uv
if !errorlevel! neq 0 ( pause & popd & exit /b !errorlevel! )

echo --------------------------------------------------
echo Updating dependencies...
echo --------------------------------------------------
echo Executing: uv pip install -U -r requirements.txt
uv pip install -U -r requirements.txt
if !errorlevel! neq 0 ( pause & popd & exit /b !errorlevel! )

echo ----------------------------------------
echo Update completed.
echo ----------------------------------------

pause

popd

popd
