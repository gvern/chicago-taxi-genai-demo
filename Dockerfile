# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt /app/

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the entire src directory into the container at /app/src
COPY src/ /app/src/

# Add the app directory to the Python path
ENV PYTHONPATH="/app"

# No ENTRYPOINT needed here if the components specify their own commands or if run via Vertex AI custom jobs
# If a specific script needs to be run by default, uncomment and adjust:
# ENTRYPOINT ["python", "/app/src/your_main_script.py"]
