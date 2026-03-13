# Stage 1: Build
FROM python:3.11-slim AS builder

# Install uv by copying it from the official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Enable bytecode compilation for faster startup
ENV UV_COMPILE_BYTECODE=1
# Use copy mode instead of hardlinks (required for Docker)
ENV UV_LINK_MODE=copy

# Copy only dependency files first to leverage Docker caching
COPY requirements.txt .

# Install dependencies into a virtualenv
# This replaces your "pip install" command
# RUN uv pip install --no-cache -r requirements.txt
RUN uv pip install --system --no-cache -r requirements.txt

# Stage 2: Runtime
FROM python:3.11-slim

WORKDIR /app

# Copy the installed packages from the builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy your source code
COPY . .

CMD ["python", "main.py"]