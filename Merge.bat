chcp 65001 > NUL
@echo off

pushd %~dp0
echo Running gradio_tabs/merge.py...
venv\Scripts\python -m gradio_tabs.merge

if %errorlevel% neq 0 ( pause & popd & exit /b %errorlevel% )

popd
pause