@echo off
echo.
echo ========================================
echo   CoachAI Full Stack Application
echo ========================================
echo.

REM Load API key from local config file (if exists)
if exist api_config.bat (
    call api_config.bat
    echo Loaded API key from api_config.bat
) else (
    echo WARNING: api_config.bat not found!
    echo Create it from api_config.bat.template to enable AI feedback
)

REM Check if GEMINI_API_KEY is set
if "%GEMINI_API_KEY%"=="" (
    echo.
    echo WARNING: GEMINI_API_KEY is not set!
    echo The app will work, but will use basic feedback instead of AI coaching.
    echo.
)

echo Starting backend server...
echo.

REM Activate conda environment and run server
call conda activate poseformerv2
python fitness_backend_server.py

pause




