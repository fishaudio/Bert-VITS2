chcp 65001 > NUL
@echo off

pushd %~dp0
echo Running gradio_tabs/dataset.py...
venv\Scripts\python -m gradio_tabs.dataset

if %errorlevel% neq 0 ( pause & popd & exit /b %errorlevel% )

popd
pause