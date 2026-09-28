# Use the official Python Linux image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file first to leverage Docker cache
COPY requirements.txt .

# Install dependencies with a massively increased timeout for large files.
# llama-cpp-python uses a prebuilt wheel when available (much faster than compiling).
RUN pip install --no-cache-dir --default-timeout=1000 \
      --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu \
      -r requirements.txt

# Copy the rest of the application code
COPY . .

# Command to run your firewall script
CMD ["python", "firewall.py"]
