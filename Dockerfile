# Use Python base image
FROM python:3.11-slim

# Set environment variables
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
RUN pip install --upgrade pip

# Copy project files
COPY ./app /app

# Install Python dependencies
RUN pip install --no-cache-dir -r /app/requirements.txt
RUN pip install fastapi uvicorn

# Expose FastAPI's default port
EXPOSE 9090

# Run FastAPI server when the container starts
ENTRYPOINT ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "9090"]
