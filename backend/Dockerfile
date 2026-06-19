FROM python:3.10-slim

# Create a non-root user for security (required by HF Spaces)
RUN useradd -m -u 1000 user

WORKDIR /app

# Install build essential for potential dependencies compile
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

# Copy the rest of the application files with user ownership
COPY --chown=user . /app

# Switch to the non-root user
USER user

# Expose the port HF Spaces expects
EXPOSE 7860

# Command to run the FastAPI app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
