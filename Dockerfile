FROM python:3.11-slim

# Prevents Python from writing .pyc files and buffers
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip
RUN pip install -r /app/requirements.txt

# Copy project
COPY . /app

# Make entrypoint executable
RUN chmod +x /app/entrypoint.sh

# Expose port (Vercel provides PORT env)
EXPOSE 8000

CMD ["/app/entrypoint.sh"]
