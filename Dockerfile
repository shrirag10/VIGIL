# ═══════════════════════════════════════════
# VIGIL V7.0 — Multi-stage Dockerfile
# Stage 1: Build React frontend
# Stage 2: Production Python + static assets
# ═══════════════════════════════════════════

# ── Stage 1: Build frontend ──
FROM node:20-alpine AS frontend-build

WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci || npm install
COPY frontend/ ./
RUN npm run build

# ── Stage 2: Production image ──
FROM python:3.11-slim

LABEL maintainer="VIGIL Team"
LABEL description="VIGIL V7.0 — Vehicle-Installed Guard for Injury Limitation"

# System dependencies for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Generate gRPC stubs
COPY proto/ proto/
RUN mkdir -p services/generated && \
    python -m grpc_tools.protoc \
    -I proto \
    --python_out=services/generated \
    --grpc_python_out=services/generated \
    proto/vigil.proto && \
    touch services/__init__.py services/generated/__init__.py && \
    sed -i 's/import vigil_pb2/from services.generated import vigil_pb2/' services/generated/vigil_pb2_grpc.py || true

# Copy application code
COPY VIGIL.py .
COPY services/ services/
RUN echo '[]' > zones.json && echo '[]' > violations.json

# Copy built frontend from stage 1
COPY --from=frontend-build /app/static/ static/

# Create directories for runtime data
RUN mkdir -p recordings reports

# Expose ports: FastAPI (8000), gRPC (50051)
EXPOSE 8000 50051

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')" || exit 1

# Run VIGIL
CMD ["python", "VIGIL.py"]
