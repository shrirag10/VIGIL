#!/usr/bin/env bash
# ═══════════════════════════════════════════
# VIGIL V7.0 — Setup & Build Script
# ═══════════════════════════════════════════
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "═══════════════════════════════════════════"
echo " VIGIL V7.0 Setup"
echo " React + FastAPI + gRPC + Kafka"
echo "═══════════════════════════════════════════"
echo ""

# ── 1. Python dependencies ──
echo "▸ Installing Python dependencies..."
pip install fastapi uvicorn pydantic opencv-python numpy ultralytics reportlab psutil 2>/dev/null || true

echo "▸ Installing gRPC dependencies..."
pip install grpcio grpcio-tools 2>/dev/null || true

echo "▸ Installing Kafka dependencies..."
pip install kafka-python 2>/dev/null || true

# ── 2. Generate gRPC stubs ──
echo ""
echo "▸ Generating gRPC protobuf stubs..."
mkdir -p services/generated
python -m grpc_tools.protoc \
    -I proto \
    --python_out=services/generated \
    --grpc_python_out=services/generated \
    proto/vigil.proto 2>/dev/null && echo "  ✓ gRPC stubs generated" || echo "  ⚠ gRPC stub generation failed (optional)"

# Fix imports in generated files (grpc_tools generates relative imports incorrectly)
if [ -f "services/generated/vigil_pb2_grpc.py" ]; then
    sed -i.bak 's/import vigil_pb2/from services.generated import vigil_pb2/' services/generated/vigil_pb2_grpc.py 2>/dev/null || true
    rm -f services/generated/vigil_pb2_grpc.py.bak 2>/dev/null || true
fi

# ── 3. Build React frontend ──
echo ""
echo "▸ Building React frontend..."
cd frontend

if ! command -v node &> /dev/null; then
    echo "  ⚠ Node.js not found. Install from https://nodejs.org/"
    echo "  Skipping frontend build — will use legacy dashboard"
    cd ..
else
    echo "  Installing npm dependencies..."
    npm install 2>/dev/null

    echo "  Building production bundle..."
    npm run build 2>/dev/null && echo "  ✓ React app built → static/" || echo "  ⚠ Build failed — will use legacy dashboard"
    cd ..
fi

# ── 4. Done ──
echo ""
echo "═══════════════════════════════════════════"
echo " Setup complete!"
echo ""
echo " Start VIGIL:"
echo "   python3 VIGIL.py"
echo ""
echo " Dashboard: http://localhost:8000"
echo " gRPC:      localhost:50051"
echo ""
echo " Optional: Start Kafka (if installed):"
echo "   brew services start kafka  # macOS"
echo "   # or: docker-compose up -d kafka"
echo "═══════════════════════════════════════════"
