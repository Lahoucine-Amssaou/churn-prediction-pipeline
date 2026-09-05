# LAYER 1: BASE IMAGE
FROM python:3.11-slim

# LAYER 2: WORKING DIRECTORY
WORKDIR /app

# LAYER 3: SYSTEM DEPENDENCIES
# --no-install-recommends keeps the image lean.
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# LAYER 4: PYTHON DEPENDENCIES
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# LAYER 5: APPLICATION CODE
# We copy the rest of the project into the container.
COPY src/ ./src/
COPY app/ ./app/
COPY artifacts/ ./artifacts/

# LAYER 6: ENVIRONMENT VARIABLES
ENV ARTIFACTS_DIR=artifacts

# We tell Python not to buffer output so logs appear immediately in Docker
ENV PYTHONUNBUFFERED=1

# We tell Python where to find the src/ module
ENV PYTHONPATH=/app/src:/app

# LAYER 7: EXPOSE PORT
EXPOSE 8000

# LAYER 8: STARTUP COMMAND
# uvicorn: the server that runs FastAPI
# app.main:app — the "app" object inside app/main.py
# --host 0.0.0.0 — listen on all network interfaces (required in containers)
# --port 8000 — the port we exposed above
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
