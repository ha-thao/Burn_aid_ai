FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for OpenCV and Pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project code, models and assets
COPY . .

# Hugging Face Spaces port
EXPOSE 7860

# Run unified server directly serving HTML and AI APIs at root
CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "7860"]
