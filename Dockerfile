# Fleet-Mind Production Docker Image
# Multi-stage build for optimized production deployment

# Build stage
FROM python:3.12-slim as builder

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt /tmp/
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r /tmp/requirements.txt

# Production stage
FROM python:3.12-slim as production

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    FLEET_MIND_ENV=production

# Install runtime system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user
RUN groupadd -r fleetmind && useradd -r -g fleetmind fleetmind

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Create application directory
WORKDIR /app

# Copy application code
COPY fleet_mind/ ./fleet_mind/
COPY scripts/ ./scripts/
COPY pyproject.toml README.md ./

# Set ownership
RUN chown -R fleetmind:fleetmind /app

# Create directories for data and logs
RUN mkdir -p /app/data /app/logs && \
    chown -R fleetmind:fleetmind /app/data /app/logs

# Switch to non-root user
USER fleetmind

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import fleet_mind; print('OK')" || exit 1

# Expose ports
EXPOSE 8080 8081 8082

# Default command
CMD ["python", "-m", "fleet_mind.cli", "--help"]

# Labels
LABEL maintainer="Daniel Schmidt <daniel@terragon.ai>" \
      version="0.1.0" \
      description="Fleet-Mind: Realtime Swarm LLM Coordination Platform" \
      org.opencontainers.image.source="https://github.com/terragon-labs/fleet-mind" \
      org.opencontainers.image.licenses="MIT"