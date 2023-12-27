@REM 改行コードがLFだとエラーが出るのでCRLFにしないといけない……

@echo off

echo Running app.py...
venv\Scripts\python app.py

if %errorlevel% neq 0 ( pause & popd & exit /b %errorlevel% )

pause