@echo off
echo ================================
echo Indian Sign Language Recognition
echo ================================
echo.
echo Starting the web application...
echo.
echo Please wait while the server starts...
echo Once you see "Running on http://127.0.0.1:5000"
echo Open your browser and go to: http://localhost:5000
echo.
echo Press Ctrl+C to stop the server
echo ================================
echo.

python web_app.py

echo.
echo ================================
echo Server stopped. Press any key to exit.
pause > nul