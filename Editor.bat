chcp 65001 > NUL
@echo off

pushd %~dp0
echo Running server_editor.py --inbrowser
venv\Scripts\python server_editor.py --inbrowser

if %errorlevel% neq 0 ( pause & popd & exit /b %errorlevel% )

popd
pause