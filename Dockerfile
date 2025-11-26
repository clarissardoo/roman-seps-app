FROM python:3.12.11-slim

# ---------------------------------------------------------
# System dependencies
# ---------------------------------------------------------
RUN apt-get update && apt-get install -y \
    build-essential \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---------------------------------------------------------
# Install core scientific stack FIRST
# ---------------------------------------------------------
RUN pip install --upgrade pip setuptools wheel

# Install numpy first to satisfy build requirements of astropy/orbitize/etc.
RUN pip install numpy

# Install cython (orbitize & radvel need this to compile extensions)
RUN pip install cython


# ---------------------------------------------------------
# Copy dependency list
# ---------------------------------------------------------
COPY requirements.txt .

# ---------------------------------------------------------
# Now install the rest
# ---------------------------------------------------------
RUN pip install --no-build-isolation -r requirements.txt

# ---------------------------------------------------------
# Copy your code
# ---------------------------------------------------------
COPY . .

EXPOSE 10000
CMD gunicorn --bind 0.0.0.0:10000 app:app