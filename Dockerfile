# Use official Python base image
FROM python:3.11

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY flask_app/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire project structure
COPY flask_app/ ./flask_app/
COPY artifacts/ ./artifacts/
COPY templates/ ./templates/
COPY static/ ./static/

# Set environment variables
ENV FLASK_APP=flask_app/app.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_ENV=development

# Expose the Flask default port
EXPOSE 5000

# Run the app
CMD ["python", "flask_app/app.py"]
