chcp 65001 > NUL

@REM https://github.com/Zuntan03/EasyBertVits2 より引用・改変

@echo off
set PS_CMD=PowerShell -Version 5.1 -ExecutionPolicy Bypass
set CURL_CMD=C:\Windows\System32\curl.exe

if not exist %CURL_CMD% (
	echo [ERROR] %CURL_CMD% が見つかりません。
	pause & exit /b 1
)

if "%1" neq "" (
	set PYTHON_DIR=%~dp0%~1
) else (
	set PYTHON_DIR=%~dp0python
)
set PYTHON_CMD=%PYTHON_DIR%\python.exe

if "%2" neq "" (
	set VENV_DIR=%~dp0%~2
) else (
	set VENV_DIR=%~dp0venv
)

echo PS_CMD: %PS_CMD%
echo CURL_CMD: %CURL_CMD%
echo PYTHON_CMD: %PYTHON_CMD%
echo VENV_DIR: %VENV_DIR%

if not exist %PYTHON_DIR%\ (
	echo https://www.python.org/
	echo %CURL_CMD% -o python.zip https://www.python.org/ftp/python/3.10.11/python-3.10.11-embed-amd64.zip
	%CURL_CMD% -o python.zip https://www.python.org/ftp/python/3.10.11/python-3.10.11-embed-amd64.zip
	if %errorlevel% neq 0 ( pause & exit /b %errorlevel% )

	echo %PS_CMD% Expand-Archive -Path python.zip -DestinationPath %PYTHON_DIR%
	%PS_CMD% Expand-Archive -Path python.zip -DestinationPath %PYTHON_DIR%
	if %errorlevel% neq 0 ( pause & exit /b %errorlevel% )

	echo del python.zip
	del python.zip
	if %errorlevel% neq 0 ( pause & exit /b %errorlevel% )

	echo %PS_CMD% "&{(Get-Content '%PYTHON_DIR%/python310._pth') -creplace '#import site', 'import site' | Set-Content '%PYTHON_DIR%/python310._pth' }"
	%PS_CMD% "&{(Get-Content '%PYTHON_DIR%/python310._pth') -creplace '#import site', 'import site' | Set-Content '%PYTHON_DIR%/python310._pth' }"
	if %errorlevel% neq 0 ( pause & exit /b %errorlevel% )

	echo https://github.com/pypa/get-pip
	echo %CURL_CMD% -o %PYTHON_DIR%\get-pip.py https://bootstrap.pypa.io/get-pip.py
	%CURL_CMD% -o %PYTHON_DIR%\get-pip.py https://bootstrap.pypa.io/get-pip.py
	if %errorlevel% neq 0 ( pause & exit /b %errorlevel% )

	echo %PYTHON_CMD% %PYTHON_DIR%\get-pip.py --no-warn-script-location
	%PYTHON_CMD% %PYTHON_DIR%\get-pip.py --no-warn-script-location
	if %errorlevel% neq 0 ( pause & exit /b %errorlevel% )

	echo %PYTHON_CMD% -m pip install virtualenv --no-warn-script-location
	%PYTHON_CMD% -m pip install virtualenv --no-warn-script-location
	if %errorlevel% neq 0 ( pause & exit /b %errorlevel% )
)

if not exist %VENV_DIR%\ (
	echo %PYTHON_CMD% -m virtualenv --copies %VENV_DIR%
	%PYTHON_CMD% -m virtualenv --copies %VENV_DIR%
	if %errorlevel% neq 0 ( pause & exit /b %errorlevel% )
)

echo call %VENV_DIR%\Scripts\activate.bat
call %VENV_DIR%\Scripts\activate.bat
if %errorlevel% neq 0 ( pause & exit /b %errorlevel% )

echo python -m pip install --upgrade pip
python -m pip install --upgrade pip
if %errorlevel% neq 0 ( pause & exit /b %errorlevel% )
