FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Only copy requirements first
COPY requirements.txt .

# Install dependencies without building binary extensions
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Expose port and run
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]