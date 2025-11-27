FROM python:3.12.11-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    git \
    git-lfs \
    && rm -rf /var/lib/apt/lists/*

RUN git lfs install

WORKDIR /app

RUN pip install --upgrade pip setuptools wheel
RUN pip install numpy
RUN pip install cython

COPY requirements.txt .
RUN pip install --no-build-isolation -r requirements.txt

# Clone the repo WITH LFS files
RUN git clone https://github.com/clarissardoo/roman-seps-app.git /tmp/repo && \
    cd /tmp/repo && \
    git lfs pull && \
    cp -r /tmp/repo/* /app/ && \
    rm -rf /tmp/repo/.git

EXPOSE 8080
CMD gunicorn --bind 0.0.0.0:8080 app:app