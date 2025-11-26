# ---------------------------------------------------------
# Use a stable Python version that supports NumPy/Astropy
# ---------------------------------------------------------
FROM python:3.12.11-slim

# ---------------------------------------------------------
# System dependencies (needed for numpy, scipy, astropy)
# ---------------------------------------------------------
RUN apt-get update && apt-get install -y \
    build-essential \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    libatlas-base-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------
# Create app directory
# ---------------------------------------------------------
WORKDIR /app

# ---------------------------------------------------------
# Copy dependency list first (for better caching)
# ---------------------------------------------------------
COPY requirements.txt .

# ---------------------------------------------------------
# Install Python dependencies
# ---------------------------------------------------------
RUN pip install --upgrade pip setuptools wheel \
    && pip install -r requirements.txt

# ---------------------------------------------------------
# Copy the full project
# ---------------------------------------------------------
COPY . .

# ---------------------------------------------------------
# Render automatically injects the $PORT env variable
# We do not need EXPOSE in Dockerfile for Render deployments.
# ---------------------------------------------------------

# ---------------------------------------------------------
# Command to start Flask app with Gunicorn
# Binds to 0.0.0.0 and uses the dynamic $PORT provided by Render
# ---------------------------------------------------------
CMD ["gunicorn", "--bind", "0.0.0.0:$PORT", "app:app"]
