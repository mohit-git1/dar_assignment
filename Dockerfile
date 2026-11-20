FROM python:3.11-slim

WORKDIR /app

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app" \
    PORT=8501

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application and data
COPY . .

# Bake Chroma vector database and the default ONNX embedding model during build
RUN python -m app.ingest

# Expose port (default 8501)
EXPOSE 8501

# Start the Streamlit application using the runtime $PORT environment variable
CMD streamlit run app/chat.py --server.port=${PORT} --server.address=0.0.0.0 --server.headless=true
