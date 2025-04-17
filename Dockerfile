# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt /app/

# Install any needed packages specified in requirements.txt
# --no-cache-dir: Disables the cache to keep the image size smaller
# --upgrade pip: Ensures pip is up-to-date
# -r requirements.txt: Installs packages listed in the file
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the local code directory into the container at /app
# This includes your src/ folder and potentially config/ if needed by the script
COPY src/ /app/src/
COPY config/ /app/config/

# Define the entrypoint for the container.
# This allows the container to execute the training script when run by Vertex AI.
# Vertex AI will pass arguments to this entrypoint.
ENTRYPOINT ["python", "-m", "src.model_training.train_xgboost_hpt"]
