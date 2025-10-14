#!/bin/sh
set -e

# Apply database migrations
echo "Running migrations..."
python manage.py migrate --noinput

# Collect static files
echo "Collecting static files..."
python manage.py collectstatic --noinput

# Start Gunicorn
echo "Starting Gunicorn..."
exec gunicorn financial_distress.wsgi:application --bind 0.0.0.0:${PORT:-8000} --workers 3 --log-file -
