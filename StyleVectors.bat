chcp 65001 > NUL
@echo off

pushd %~dp0
echo Running gradio_tabs/style_vectors.py...
venv\Scripts\python -m gradio_tabs.style_vectors

if %errorlevel% neq 0 ( pause & popd & exit /b %errorlevel% )

popd
pause