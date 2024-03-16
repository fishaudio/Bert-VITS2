chcp 65001 > NUL
@echo off

pushd %~dp0

set PATH=%~dp0lib\MinGit\cmd;%PATH%

pushd Style-Bert-VITS2

echo --------------------------------------------------
echo Checking Git Installation...
echo --------------------------------------------------
git --version
if %errorlevel% neq 0 ( pause & popd & exit /b %errorlevel% )

pause

echo --------------------------------------------------
echo Git pull...
echo --------------------------------------------------
git pull
if %errorlevel% neq 0 ( pause & popd & exit /b %errorlevel% )

@REM 仮想環境のpip requirements.txtを更新

echo --------------------------------------------------
echo Updating dependencies...
echo --------------------------------------------------

@REM 仮想環境を有効化
call .\venv\Scripts\activate.bat
if %errorlevel% neq 0 ( pause & popd & exit /b %errorlevel% )

pip install -U -r requirements.txt
if %errorlevel% neq 0 ( pause & popd & exit /b %errorlevel% )

echo ----------------------------------------
echo Update completed.
echo ----------------------------------------

pause

popd

popd
