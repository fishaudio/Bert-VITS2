chcp 65001 > NUL

@echo off

pushd %~dp0

echo Running webui_merge.py...
venv\Scripts\python webui_merge.py

if %errorlevel% neq 0 ( pause & popd & exit /b %errorlevel% )

popd
pause