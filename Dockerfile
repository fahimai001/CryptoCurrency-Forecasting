FROM python:3.9-slim
# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY . /app

# Install the required packages
RUN pip install -r requirements.txt

EXPOSE 5000

CMD ["python", "app.py"]