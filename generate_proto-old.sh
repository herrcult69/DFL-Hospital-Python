#!/usr/bin/env bash
# Generate Python gRPC stubs from the proto definition.
# Must be run from the project root.
set -euo pipefail

PROTO_DIR="protos"
OUT_DIR="protos"

python -m grpc_tools.protoc \
    -I"$PROTO_DIR" \
    --python_out="$OUT_DIR" \
    --grpc_python_out="$OUT_DIR" \
    "$PROTO_DIR/ring_transfer.proto"

# Fix relative imports in the generated grpc file
sed -i 's/import ring_transfer_pb2/from protos import ring_transfer_pb2/' \
    "$OUT_DIR/ring_transfer_pb2_grpc.py"

echo "gRPC stubs generated in $OUT_DIR/"