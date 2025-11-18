FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    apparmor \
    apparmor-utils \
    auditd \
    python3-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash ucognet

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY pyproject.toml poetry.lock ./
RUN pip install poetry && poetry config virtualenvs.create false && poetry install --only main

# Copy source code with read-only permissions
COPY --chown=ucognet:ucognet src/ ./src/
COPY --chown=ucognet:ucognet *.py ./

# Make source code read-only
RUN chmod -R 444 src/ *.py && \
    find . -name "*.py" -exec chmod 544 {} \;

# Create data and logs directories with write permissions
RUN mkdir -p /data /logs && \
    chown -R ucognet:ucognet /data /logs

# Switch to non-root user
USER ucognet

# Set environment variables for security
ENV PYTHONPATH=/app/src
ENV UCogNet_SANDBOX=true
ENV UCogNet_READONLY=true

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import sys; sys.exit(0)" || exit 1

# Default command
CMD ["python3", "ucognet_complete_system.py"]