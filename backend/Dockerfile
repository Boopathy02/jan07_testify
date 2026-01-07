# ============================================================
# Base image (Python 3.11 – stable, PyYAML-safe)
# ============================================================
FROM python:3.11-slim-bookworm

# ============================================================
# Environment
# ============================================================
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DISPLAY=:99 \
    TOKENIZERS_PARALLELISM=false \
    JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64 \
    PATH=/usr/lib/jvm/java-17-openjdk-amd64/bin:$PATH

WORKDIR /app

# ============================================================
# System dependencies
# ============================================================
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    ca-certificates \
    unzip \
    gnupg \
    net-tools \
    \
    # Java (Allure)
    openjdk-17-jdk \
    \
    # Playwright / Chromium deps
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libffi-dev \
    libgl1 \
    libnss3 \
    libatk-bridge2.0-0 \
    libx11-xcb1 \
    libdrm2 \
    libxcomposite1 \
    libxdamage1 \
    libxrandr2 \
    libgbm1 \
    libasound2 \
    libpangocairo-1.0-0 \
    libgtk-3-0 \
    fonts-liberation \
    \
    # OCR
    tesseract-ocr \
    \
    # VNC / noVNC stack
    xvfb \
    x11vnc \
    fluxbox \
    novnc \
    websockify \
    \
    && rm -rf /var/lib/apt/lists/*

# ============================================================
# Allure CLI (ZIP install – fail fast)
# ============================================================
RUN set -eux; \
    ALLURE_VERSION=2.36.0; \
    curl -fSL \
      "https://repo.maven.apache.org/maven2/io/qameta/allure/allure-commandline/${ALLURE_VERSION}/allure-commandline-${ALLURE_VERSION}.zip" \
      -o /tmp/allure.zip; \
    unzip /tmp/allure.zip -d /opt; \
    ln -sf /opt/allure-${ALLURE_VERSION}/bin/allure /usr/bin/allure; \
    rm -f /tmp/allure.zip; \
    allure --version

# ============================================================
# Python dependencies
# ============================================================
COPY requirements.txt .

RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir uvicorn "playwright==1.52.0" && \
    python -m playwright install --with-deps chromium

# ============================================================
# Application source
# ============================================================
COPY . .

# noVNC root fix
RUN ln -sf /usr/share/novnc/vnc.html /usr/share/novnc/index.html

# ============================================================
# Network
# ============================================================
EXPOSE 8001 6080 5900

# ============================================================
# Startup (FIXED – 0.0.0.0 BINDING)
# ============================================================
CMD bash -c "\
  echo 'Starting Xvfb...' && \
  Xvfb :99 -screen 0 1920x1080x24 & \
  \
  echo 'Waiting for X display...' && \
  until xdpyinfo -display :99 >/dev/null 2>&1; do sleep 1; done && \
  \
  echo 'Starting Fluxbox...' && \
  fluxbox & \
  \
  echo 'Starting x11vnc...' && \
  x11vnc -display :99 -nopw -forever -shared \
         -listen 0.0.0.0 -rfbport 5900 & \
  \
  echo 'Starting noVNC...' && \
  websockify --web=/usr/share/novnc/ \
             --wrap-mode=ignore \
             0.0.0.0:6080 0.0.0.0:5900 & \
  \
  echo 'Starting backend API...' && \
  python -m uvicorn main:app --host 0.0.0.0 --port 8001"

