FROM python:3.9-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FLASK_APP=app.py \
    FLASK_ENV=production \
    PYTHONPATH=/app

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    wget \
    curl \
    unzip \
    libxml2-dev \
    libxslt1-dev \
    zlib1g-dev \
    libjpeg-dev \
    libpng-dev \
    libfreetype6-dev \
    liblcms2-dev \
    libwebp-dev \
    tcl8.6-dev \
    tk8.6-dev \
    python3-tk \
    libharfbuzz-dev \
    libfribidi-dev \
    libxcb1-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -r appuser && useradd -r -g appuser appuser

RUN mkdir -p /app/logs /app/uploads /tmp/resume-uploads /var/log/app && \
    chown -R appuser:appuser /app /tmp/resume-uploads /var/log/app

COPY requirements.txt /app/

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . /app/

RUN mkdir -p /app/skillScan/models /app/skillScan/data /app/templates /app/static

RUN chown -R appuser:appuser /app && \
    chmod -R 755 /app

USER appuser

RUN echo '#!/bin/bash\n\
echo "Starting Resume Analysis Application..."\n\
echo "Python version: $(python --version)"\n\
echo "Working directory: $(pwd)"\n\
echo "Environment: $FLASK_ENV"\n\
echo "Available disk space:"\n\
df -h /tmp\n\
echo "Starting Flask application..."\n\
exec python app.py' > /app/start.sh && \
chmod +x /app/start.sh

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/ || exit 1

EXPOSE 5000

VOLUME ["/tmp/resume-uploads", "/var/log/app"]

LABEL maintainer="durgeshsingh12712@gmail.com" \
      version="0.0.1.0" \
      description="SkillScan AI Resume-Analyzer Flask Application" \
      org.opencontainers.image.source="https://github.com/Durgeshsingh12712/End-to-End-SkillScan-AI-Resume-Analyzer"

CMD ["/app/start.sh"]