@echo off

echo Running webui_style_vectors.py...
venv\Scripts\python webui_style_vectors.py

if %errorlevel% neq 0 ( pause & popd & exit /b %errorlevel% )

pause