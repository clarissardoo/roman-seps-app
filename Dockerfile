# ---------------------------------------------------------
# Use Python version compatible with NumPy/Astropy
# ---------------------------------------------------------
FROM python:3.12.11-slim

# ---------------------------------------------------------
# System dependencies (needed for numpy, scipy, astropy, radvel, orbitize)
# ---------------------------------------------------------
RUN apt-get update && apt-get install -y \
    build-essential \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------
# Create app directory
# ---------------------------------------------------------
WORKDIR /app

# ---------------------------------------------------------
# Copy dependency list first (for caching)
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
# Start Flask app with Gunicorn
# Render provides $PORT automatically.
# ---------------------------------------------------------
CMD ["gunicorn", "--bind", "0.0.0.0:$PORT", "app:app"]
