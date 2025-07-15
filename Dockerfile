FROM python:3.8-slim-bullseye

FROM python:3.9-slim-bullseye

FROM python:3.10-slim-bullseye

WORKDIR /app

COPY . /app

RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 5000

CMD ["python3", "app.py"]