@echo off
echo Starting MT Provider Selection Engine...
echo Open http://localhost:8502 in your browser
echo (Backend API starts automatically inside the app)
echo.
.venv\Scripts\python.exe -m streamlit run streamlit_app.py --server.port 8502
