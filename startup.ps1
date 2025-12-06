# ============================================================================
# BEHAVIOR DETECTION SYSTEM - STARTUP SCRIPT (PowerShell)
# ============================================================================
# This script sets up and runs the behavior detection system
# ============================================================================

$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot

Write-Host ""
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host "          AI-Powered Behavior Detection System v1.0.0" -ForegroundColor Cyan
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host ""

# Check if virtual environment exists
if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Failed to create virtual environment" -ForegroundColor Red
        Write-Host "Make sure Python 3.10+ is installed and in PATH" -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& .\.venv\Scripts\Activate.ps1
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to activate virtual environment" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Upgrade pip
Write-Host "Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip --quiet

# Install requirements
Write-Host "Installing dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to install dependencies" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""
Write-Host "============================================================================" -ForegroundColor Green
Write-Host "Installation complete!" -ForegroundColor Green
Write-Host "============================================================================" -ForegroundColor Green
Write-Host ""

# Menu loop
$continue = $true
while ($continue) {
    Write-Host "What would you like to do?" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "1 - Test object detection on webcam (Phase 1)"
    Write-Host "2 - Test behavior detection on webcam (Phase 2)"
    Write-Host "3 - Process a video file"
    Write-Host "4 - View quick start guide"
    Write-Host "5 - Run tests"
    Write-Host "6 - Exit"
    Write-Host ""
    
    $choice = Read-Host "Enter your choice (1-6)"
    
    switch ($choice) {
        "1" {
            Write-Host ""
            Write-Host "Starting object detection on webcam..." -ForegroundColor Yellow
            Write-Host "Press 'q' to quit." -ForegroundColor Yellow
            Write-Host ""
            python yolo_object_detection/main.py --source 0 --show
            Write-Host ""
        }
        
        "2" {
            Write-Host ""
            Write-Host "Starting behavior detection on webcam..." -ForegroundColor Yellow
            Write-Host "Press 'q' to quit." -ForegroundColor Yellow
            Write-Host ""
            python run_behaviour.py --source 0 --show --events-csv runs/events.csv
            Write-Host ""
            Write-Host "Check runs/events.csv for detected behaviors" -ForegroundColor Green
            Write-Host ""
        }
        
        "3" {
            Write-Host ""
            $video = Read-Host "Enter path to video file"
            if (-not (Test-Path $video)) {
                Write-Host "File not found: $video" -ForegroundColor Red
                continue
            }
            Write-Host ""
            Write-Host "Processing video: $video" -ForegroundColor Yellow
            python run_behaviour.py --source "$video" --show --events-csv runs/video_events.csv --save-dir runs/video_output
            Write-Host ""
            Write-Host "Results saved to runs/video_output" -ForegroundColor Green
            Write-Host "Events saved to runs/video_events.csv" -ForegroundColor Green
            Write-Host ""
        }
        
        "4" {
            Write-Host ""
            Write-Host "Displaying quick start guide..." -ForegroundColor Yellow
            Write-Host ""
            python QUICKSTART.py
            Write-Host ""
        }
        
        "5" {
            Write-Host ""
            Write-Host "Running tests..." -ForegroundColor Yellow
            Write-Host ""
            python -m unittest test_behavior_detection -v
            Read-Host "Press Enter to continue"
            Write-Host ""
        }
        
        "6" {
            Write-Host ""
            Write-Host "Exiting..." -ForegroundColor Yellow
            $continue = $false
        }
        
        default {
            Write-Host "Invalid choice. Please try again." -ForegroundColor Red
            Write-Host ""
        }
    }
}
