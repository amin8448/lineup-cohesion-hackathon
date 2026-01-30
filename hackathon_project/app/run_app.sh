#!/bin/bash
# Launch script for Lineup Cohesion Analyzer

echo "======================================"
echo "Lineup Cohesion Analyzer"
echo "NEU Sports Analytics Hackathon 2026"
echo "======================================"

# Check if data exists
if [ ! -f "../data/cohesion_results.csv" ]; then
    echo "ERROR: Data file not found!"
    echo "Please run the analysis scripts first:"
    echo "  cd ../notebooks && python 03_full_analysis.py"
    exit 1
fi

echo "Starting Streamlit server..."
echo "Open http://localhost:8501 in your browser"
echo ""

streamlit run streamlit_app.py --server.port 8501
