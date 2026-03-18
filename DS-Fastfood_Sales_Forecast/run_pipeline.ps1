# Sales Forecast Pipeline Runner
# Usage: .\run_pipeline.ps1

Write-Host "Starting Fast Food Sales Forecast Pipeline..." -ForegroundColor Cyan

# Set PYTHONPATH to handle local 'src' module
$env:PYTHONPATH = "."

Write-Host "1. Running ETL Pipeline..." -ForegroundColor Yellow
python src/etl.py

Write-Host "2. Running Feature Engineering Pipeline..." -ForegroundColor Yellow
python src/features.py

Write-Host "3. Running Model Training..." -ForegroundColor Yellow
python src/train.py

Write-Host "4. Launching Streamlit Dashboard..." -ForegroundColor Green
streamlit run app.py
