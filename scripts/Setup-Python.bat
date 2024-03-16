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

echo --------------------------------------------------
echo PS_CMD: %PS_CMD%
echo CURL_CMD: %CURL_CMD%
echo PYTHON_CMD: %PYTHON_CMD%
echo PYTHON_DIR: %PYTHON_DIR%
echo VENV_DIR: %VENV_DIR%
echo --------------------------------------------------

pause

if not exist "%PYTHON_DIR%"\ (
	echo --------------------------------------------------
	echo Downloading Python...
	echo --------------------------------------------------
	echo Executing: %CURL_CMD% -o python.zip https://www.python.org/ftp/python/3.10.11/python-3.10.11-embed-amd64.zip
	pause
	%CURL_CMD% -o python.zip https://www.python.org/ftp/python/3.10.11/python-3.10.11-embed-amd64.zip
	if %errorlevel% neq 0 ( pause & exit /b %errorlevel% )

	echo --------------------------------------------------
	echo Extracting zip...
	echo --------------------------------------------------
	echo Executing: %PS_CMD% Expand-Archive -Path python.zip -DestinationPath \"%PYTHON_DIR%\"
	pause
	%PS_CMD% Expand-Archive -Path python.zip -DestinationPath \"%PYTHON_DIR%\"
	if %errorlevel% neq 0 ( pause & exit /b %errorlevel% )

	echo --------------------------------------------------
	echo Removing python.zip...
	echo --------------------------------------------------
	echo Executing: del python.zip
	pause
	del python.zip
	if %errorlevel% neq 0 ( pause & exit /b %errorlevel% )

	echo --------------------------------------------------
	echo Enabling 'site' module in the embedded Python environment...
	echo --------------------------------------------------
	echo Executing: %PS_CMD% "&{(Get-Content '%PYTHON_DIR%/python310._pth') -creplace '#import site', 'import site' | Set-Content '%PYTHON_DIR%/python310._pth' }"
	pause
	%PS_CMD% "&{(Get-Content '%PYTHON_DIR%/python310._pth') -creplace '#import site', 'import site' | Set-Content '%PYTHON_DIR%/python310._pth' }"
	if %errorlevel% neq 0 ( pause & exit /b %errorlevel% )

	echo --------------------------------------------------
	echo Installing pip and virtualenv...
	echo --------------------------------------------------
	echo Executing: %CURL_CMD% -o "%PYTHON_DIR%\get-pip.py" https://bootstrap.pypa.io/get-pip.py
	pause
	%CURL_CMD% -o "%PYTHON_DIR%\get-pip.py" https://bootstrap.pypa.io/get-pip.py
	if %errorlevel% neq 0 ( pause & exit /b %errorlevel% )

	echo --------------------------------------------------
	echo Installing pip...
	echo --------------------------------------------------
	echo Executing: "%PYTHON_CMD%" "%PYTHON_DIR%\get-pip.py" --no-warn-script-location
	pause
	"%PYTHON_CMD%" "%PYTHON_DIR%\get-pip.py" --no-warn-script-location
	if %errorlevel% neq 0 ( pause & exit /b %errorlevel% )

	echo --------------------------------------------------
	echo Installing virtualenv...
	echo --------------------------------------------------
	echo Executing: "%PYTHON_CMD%" -m pip install virtualenv --no-warn-script-location
	pause
	"%PYTHON_CMD%" -m pip install virtualenv --no-warn-script-location
	if %errorlevel% neq 0 ( pause & exit /b %errorlevel% )
)

if not exist %VENV_DIR%\ (
	echo --------------------------------------------------
	echo Creating virtual environment...
	echo --------------------------------------------------
	echo Executing: "%PYTHON_CMD%" -m virtualenv --copies "%VENV_DIR%"
	pause
	"%PYTHON_CMD%" -m virtualenv --copies "%VENV_DIR%"
	if %errorlevel% neq 0 ( pause & exit /b %errorlevel% )
)

echo --------------------------------------------------
echo Actibating virtual environment...
echo --------------------------------------------------
echo Executing: call "%VENV_DIR%\Scripts\activate.bat"
pause
call "%VENV_DIR%\Scripts\activate.bat"
if %errorlevel% neq 0 ( pause & exit /b %errorlevel% )

echo --------------------------------------------------
echo Upgrading pip...
echo --------------------------------------------------
echo Executing: python -m pip install --upgrade pip
pause
python -m pip install --upgrade pip
if %errorlevel% neq 0 ( pause & exit /b %errorlevel% )

echo --------------------------------------------------
echo Completed.
echo --------------------------------------------------

pause