# #```

# #
# ==========================================
# Stage 1: Builder (dependencies)
# ==========================================
FROM python:3.10-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*
# App directory (think as cd command)
WORKDIR /build

# Each command -> layer, Layer caching: If any command changes, re-run from that layer,else cache


# Copy only requirements (cache layer)
COPY requirements.txt .

# Install to /install for copying to final stage
RUN pip install --user --no-cache-dir -r requirements.txt

# Download spaCy model
RUN python -m spacy download en_core_web_sm
# Add dockerignore -> ignore unnecessary files (.env/heavy node modules)
# ==========================================
# Stage 2: Runtime (minimal)
# ==========================================
FROM python:3.10-slim
# Run command is used to install packages in the image
# Create non-root user
RUN useradd -m -u 1000 raguser && \
    mkdir -p /app /data /output && \
    chown -R raguser:raguser /app /data /output

# Copy Python packages from builder
COPY --from=builder --chown=raguser:raguser /root/.local /home/raguser/.local

# Set PATH for user-installed packages
ENV PATH=/home/raguser/.local/bin:$PATH

WORKDIR /app

# Copy application code
COPY --chown=raguser:raguser . .

# Switch to non-root user
USER raguser

# Expose ports (if needed for API)
EXPOSE 8000


# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"
# Cmd (not run) -> start container with this command 
# Default command
CMD ["python", "-m", "src.orchestrator"]

