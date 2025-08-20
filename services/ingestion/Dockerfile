FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y gcc g++ && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/logs /app/src

# Ensure __init__.py exists
RUN touch /app/src/__init__.py

# Set PYTHONPATH
ENV PYTHONPATH=/app

# Run the scheduler
CMD ["python", "main.py"]