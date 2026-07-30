@echo off
echo ============================================
echo   VitalWatch ER - AI Triage Monitor
echo ============================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH.
    echo Please install Python 3.10+ from https://python.org
    pause
    exit /b 1
)

REM Install dependencies if needed
echo [1/3] Checking dependencies...
pip install -r requirements.txt --quiet

echo.
echo [2/3] Training ML model (if not already trained)...
if not exist "models\best_model.joblib" (
    echo    Training models... this may take a minute.
    python src\model_training.py
    python src\model_evaluation.py
) else (
    echo    Model already trained. Skipping.
)

echo.
echo [3/3] Launching VitalWatch ER Dashboard...
echo    Opening in your browser...
echo.
streamlit run dashboard\app.py --server.headless true

pause
