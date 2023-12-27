chcp 65001 > NUL
@echo off

pushd %~dp0
echo Running webui_dataset.py...
venv\Scripts\python webui_dataset.py

if %errorlevel% neq 0 ( pause & popd & exit /b %errorlevel% )

popd
pause