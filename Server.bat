chcp 65001 > NUL
@echo off

pushd %~dp0
echo Running server_fastapi.py
venv\Scripts\python server_fastapi.py

if %errorlevel% neq 0 ( pause & popd & exit /b %errorlevel% )

popd
pause