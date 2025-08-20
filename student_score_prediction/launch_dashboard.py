#!/usr/bin/env python3
"""
Student Score Prediction Dashboard Launcher
Reliable hosting script for the Streamlit application
"""

import subprocess
import sys
import os
import time
import webbrowser
from pathlib import Path

def check_dependencies():
    """Check if all required packages are installed"""
    required_packages = [
        'streamlit', 'pandas', 'numpy', 'scikit-learn', 
        'matplotlib', 'plotly', 'seaborn', 'statsmodels'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("Installing missing packages...")
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install'
        ] + missing_packages)

def start_dashboard():
    """Start the Streamlit dashboard with proper configuration"""
    print("🚀 Starting Student Score Prediction Dashboard...")
    print("📊 Loading interactive web application...")
    
    # Ensure we're in the right directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    # Check dependencies
    check_dependencies()
    
    # Start Streamlit with proper configuration
    cmd = [
        sys.executable, '-m', 'streamlit', 'run', 
        'interactive_dashboard.py',
        '--server.headless', 'true',
        '--server.port', '8501',
        '--server.address', '0.0.0.0',
        '--server.enableCORS', 'false',
        '--server.enableXsrfProtection', 'false'
    ]
    
    print("🌐 Dashboard will be available at:")
    print("   - Local: http://localhost:8501")
    print("   - Network: http://127.0.0.1:8501")
    print("📱 Opening browser in 3 seconds...")
    
    # Start the process
    process = subprocess.Popen(cmd)
    
    # Wait a moment then open browser
    time.sleep(3)
    try:
        webbrowser.open('http://localhost:8501')
    except Exception:
        print("💡 Manually open: http://localhost:8501")
    
    try:
        # Wait for the process
        process.wait()
    except KeyboardInterrupt:
        print("\n🛑 Stopping dashboard...")
        process.terminate()

if __name__ == "__main__":
    start_dashboard()
