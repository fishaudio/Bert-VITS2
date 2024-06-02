chcp 65001 > NUL
@echo off

pushd %~dp0
echo Running initialize.py...
venv\Scripts\python initialize.py

if %errorlevel% neq 0 ( pause & popd & exit /b %errorlevel% )

popd
pause