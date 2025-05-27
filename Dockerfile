FROM python:3.9-slim

WORKDIR /app

COPY flask_app/requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r /app/requirements.txt

RUN pip install gunicorn

COPY flask_app /app/flask_app

COPY artifacts /app/artifacts

COPY templates /app/templates

COPY static /app/static

WORKDIR /app/flask_app

EXPOSE 5000

CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:app"]