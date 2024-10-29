# Stage 1: Build dependencies
FROM python:3.9-slim AS builder

# Set up working directory
WORKDIR /app

# Copy requirements.txt and install dependencies with pip (no cache)
COPY requirements.txt .
RUN apt-get update && apt-get install -y --no-install-recommends \
       build-essential \
       && pip install --no-cache-dir -r requirements.txt \
       && apt-get purge -y --auto-remove build-essential \
       && rm -rf /var/lib/apt/lists/*

# Stage 2: Final image with only the application and dependencies
FROM python:3.9-slim

# Set up working directory
WORKDIR /app

# Copy installed dependencies from the builder stage
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY . .

# Expose the application port
EXPOSE 7860

# Set environment variables for Flask
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0

# Run the Flask application with gunicorn for production-like stability
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "app:app"]
