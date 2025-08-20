@echo off
echo Starting Student Score Prediction Dashboard...
echo.
echo Dashboard will open in your default web browser.
echo Press Ctrl+C to stop the dashboard.
echo.
cd /d "%~dp0"
echo Installing required packages...
pip install streamlit plotly pandas numpy scikit-learn matplotlib seaborn statsmodels
echo.
echo Starting dashboard...
streamlit run interactive_dashboard.py --server.headless true --server.port 8501 --server.address 0.0.0.0
pause
