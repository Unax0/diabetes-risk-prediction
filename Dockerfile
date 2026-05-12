FROM python:3.11-slim

WORKDIR /app

# System libraries needed by xgboost / shap / matplotlib.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first so the layer is cached when only source changes.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code and model artifact.
COPY app.py ./
COPY src/ ./src/
COPY xgb_model_reduced.pkl ./

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", \
            "--server.port=8501", \
            "--server.address=0.0.0.0", \
            "--browser.gatherUsageStats=false"]
