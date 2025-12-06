@echo off
REM ============================================================================
REM BEHAVIOR DETECTION SYSTEM - STARTUP SCRIPT (Windows)
REM ============================================================================
REM This script sets up and runs the behavior detection system
REM ============================================================================

setlocal enabledelayedexpansion
cd /d "%~dp0"

echo.
echo ============================================================================
echo          AI-Powered Behavior Detection System v1.0.0
echo ============================================================================
echo.

REM Check if virtual environment exists
if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment
        echo Make sure Python 3.10+ is installed and in PATH
        pause
        exit /b 1
    )
)

REM Activate virtual environment
echo Activating virtual environment...
call .venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip --quiet

REM Install requirements
echo Installing dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo ============================================================================
echo Installation complete!
echo ============================================================================
echo.

REM Ask user what to do next
:menu
echo What would you like to do?
echo.
echo 1 - Test object detection on webcam (Phase 1)
echo 2 - Test behavior detection on webcam (Phase 2)
echo 3 - Process a video file
echo 4 - View documentation
echo 5 - Run tests
echo 6 - Exit
echo.

set /p choice="Enter your choice (1-6): "

if "%choice%"=="1" (
    echo.
    echo Starting object detection on webcam...
    echo Press 'q' to quit.
    echo.
    python yolo_object_detection/main.py --source 0 --show
    goto menu
)

if "%choice%"=="2" (
    echo.
    echo Starting behavior detection on webcam...
    echo Press 'q' to quit.
    echo.
    python run_behaviour.py --source 0 --show --events-csv runs/events.csv
    echo.
    echo Check runs/events.csv for detected behaviors
    goto menu
)

if "%choice%"=="3" (
    echo.
    set /p video="Enter path to video file: "
    if not exist "!video!" (
        echo File not found: !video!
        goto menu
    )
    echo.
    echo Processing video: !video!
    python run_behaviour.py --source "!video!" --show --events-csv runs/video_events.csv --save-dir runs/video_output
    echo.
    echo Results saved to runs/video_output
    echo Events saved to runs/video_events.csv
    goto menu
)

if "%choice%"=="4" (
    echo.
    echo Opening documentation...
    echo.
    python README.md
    goto menu
)

if "%choice%"=="5" (
    echo.
    echo Running tests...
    echo.
    python -m unittest test_behavior_detection -v
    pause
    goto menu
)

if "%choice%"=="6" (
    echo.
    echo Exiting...
    exit /b 0
)

echo Invalid choice. Please try again.
goto menu
