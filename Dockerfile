FROM python:3.10-slim

WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose port 7860 (Hugging Face default)
EXPOSE 7860

# CMD to start the FastAPI app using the dynamic PORT environment variable (default to 7860 for Hugging Face)
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}

