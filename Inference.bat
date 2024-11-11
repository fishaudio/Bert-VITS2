chcp 65001 > NUL
@echo off

pushd %~dp0
echo Running gradio_tabs/inference.py...
venv\Scripts\python -m gradio_tabs.inference

if %errorlevel% neq 0 ( pause & popd & exit /b %errorlevel% )

popd
pause