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
	pause
	set "PATH=%~dp0lib\PortableGit\bin;%PATH%"
	if !errorlevel! neq 0 ( pause & popd & exit /b !errorlevel! )

	echo --------------------------------------------------
	echo Checking Git Installation...
	echo --------------------------------------------------
	echo Executing: git --version
	pause
	git --version
	if !errorlevel! neq 0 ( pause & popd & exit /b !errorlevel! )
)

pause

echo --------------------------------------------------
echo Git pull...
echo --------------------------------------------------
git pull
if !errorlevel! neq 0 ( pause & popd & exit /b !errorlevel! )

@REM 仮想環境のpip requirements.txtを更新

echo --------------------------------------------------
echo Updating dependencies...
echo --------------------------------------------------

@REM 仮想環境を有効化
call .\venv\Scripts\activate.bat
if !errorlevel! neq 0 ( pause & popd & exit /b !errorlevel! )

pip install -U -r requirements.txt
if !errorlevel! neq 0 ( pause & popd & exit /b !errorlevel! )

echo ----------------------------------------
echo Update completed.
echo ----------------------------------------

pause

popd

popd
