#!/bin/bash
#
# convert_to_nb.sh — Full Acuity/Pegasus pipeline: ONNX → .nb for AMB Pro2
#
# This script runs the 3-step Acuity conversion:
#   1. Import:   ONNX → Acuity internal format (.json + .data)
#   2. Quantize: float32 → uint8 (asymmetric affine) using calibration images
#   3. Export:   Generate the .nb binary for the AMB Pro2 NPU
#
# Prerequisites:
#   - Acuity toolchain installed (set ACUITY_PATH) OR running inside Docker
#   - ONNX model + inputmeta.yml + dataset.txt in the workspace directory
#
# Usage:
#   ./convert_to_nb.sh                                       # uses defaults
#   ./convert_to_nb.sh --name best --workspace ./conversion_workspace
#   ./convert_to_nb.sh --name best --workspace ./workspace --qtype uint8
#
# The output .nb file will be at: <workspace>/output/<name>.nb
#

set -e

# ======================== DEFAULTS ========================
NAME="best"
WORKSPACE=""
QTYPE="uint8"
IMGSZ=416
INPUT_NAME="images"
OUTPUT_NAME="output0"
SKIP_IMPORT=0
SKIP_QUANTIZE=0

# ======================== PARSE ARGS ========================
while [[ $# -gt 0 ]]; do
    case "$1" in
        --name)       NAME="$2"; shift 2;;
        --workspace)  WORKSPACE="$2"; shift 2;;
        --qtype)      QTYPE="$2"; shift 2;;
        --imgsz)      IMGSZ="$2"; shift 2;;
        --input)      INPUT_NAME="$2"; shift 2;;
        --output)     OUTPUT_NAME="$2"; shift 2;;
        --skip-import)    SKIP_IMPORT=1; shift;;
        --skip-quantize)  SKIP_QUANTIZE=1; shift;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --name NAME           Model name (default: best)"
            echo "  --workspace DIR       Workspace directory containing ONNX + inputmeta"
            echo "  --qtype TYPE          Quantization type: uint8, int16, bf16 (default: uint8)"
            echo "  --imgsz SIZE          Input image size (default: 416)"
            echo "  --input NAME          ONNX input tensor name (default: images)"
            echo "  --output NAME         ONNX output tensor name (default: output0)"
            echo "  --skip-import         Skip the import step (reuse existing .json/.data)"
            echo "  --skip-quantize       Skip quantization (export float model)"
            echo "  -h, --help            Show this help"
            exit 0
            ;;
        *)
            echo "[ERROR] Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Default workspace
if [ -z "$WORKSPACE" ]; then
    WORKSPACE="$(dirname "$(readlink -f "$0")")/conversion_workspace"
fi

WORKSPACE="$(readlink -f "$WORKSPACE")"

# ======================== DETECT PEGASUS ========================
# Support both Docker (pegasus in PATH) and native (ACUITY_PATH) installations
if command -v pegasus &>/dev/null; then
    PEGASUS="pegasus"
    echo "[INFO] Using pegasus from PATH (Docker mode)"
elif [ -n "$ACUITY_PATH" ]; then
    if [ -e "$ACUITY_PATH/pegasus" ]; then
        PEGASUS="$ACUITY_PATH/pegasus"
    elif [ -e "$ACUITY_PATH/pegasus.py" ]; then
        PEGASUS="python3 $ACUITY_PATH/pegasus.py"
    else
        echo "[ERROR] pegasus not found in ACUITY_PATH=$ACUITY_PATH"
        exit 1
    fi
    echo "[INFO] Using pegasus from ACUITY_PATH=$ACUITY_PATH"
else
    echo "[ERROR] Cannot find pegasus."
    echo "        Set ACUITY_PATH or run inside the Acuity Docker container."
    exit 1
fi

# ======================== VALIDATE INPUT ========================
ONNX_FILE="$WORKSPACE/${NAME}.onnx"
INPUTMETA="$WORKSPACE/${NAME}_inputmeta.yml"

if [ ! -f "$ONNX_FILE" ]; then
    echo "[ERROR] ONNX model not found: $ONNX_FILE"
    echo "        Run export_to_onnx.py first to generate it."
    exit 1
fi

echo ""
echo "======================================================================="
echo "  YOLO26 ONNX → AMB Pro2 .nb Conversion"
echo "======================================================================="
echo "  Model name:    $NAME"
echo "  Workspace:     $WORKSPACE"
echo "  ONNX file:     $ONNX_FILE"
echo "  Quantization:  $QTYPE"
echo "  Input size:    ${IMGSZ}x${IMGSZ}"
echo "  Input tensor:  $INPUT_NAME"
echo "  Output tensor: $OUTPUT_NAME"
echo "======================================================================="
echo ""

cd "$WORKSPACE"

# ======================== STEP 1: IMPORT ========================
if [ "$SKIP_IMPORT" -eq 0 ]; then
    echo "--- Step 1/3: Import ONNX model ---"

    # Clean up previous import artifacts
    rm -f "${NAME}.json" "${NAME}.data"

    # Try the newer pegasus CLI syntax first (Docker), fall back to older syntax
    if $PEGASUS import onnx --help 2>&1 | grep -q "model-name"; then
        # Newer syntax (Docker-based Acuity)
        echo "[INFO] Using new-style pegasus import"
        $PEGASUS import onnx \
            --model ${NAME}.onnx \
            --output-model ${NAME}.json \
            --output-data ${NAME}.data
    else
        # Older syntax (native Acuity)
        echo "[INFO] Using classic-style pegasus import"
        $PEGASUS import onnx \
            --model ${NAME}.onnx \
            --output-model ${NAME}.json \
            --output-data ${NAME}.data
    fi

    if [ ! -f "${NAME}.json" ] || [ ! -f "${NAME}.data" ]; then
        echo "[ERROR] Import failed — ${NAME}.json or ${NAME}.data not generated."
        exit 1
    fi

    # Generate inputmeta if not present
    if [ ! -f "$INPUTMETA" ]; then
        echo "[INFO] Generating inputmeta..."
        $PEGASUS generate inputmeta \
            --model ${NAME}.json \
            --separated-database \
            --input-meta-output ${NAME}_inputmeta.yml
        echo "[WARN] Auto-generated inputmeta — please review ${NAME}_inputmeta.yml"
    fi

    # Generate postprocess file
    if [ ! -f "${NAME}_postprocess_file.yml" ]; then
        $PEGASUS generate postprocess-file \
            --model ${NAME}.json \
            --postprocess-file-output ${NAME}_postprocess_file.yml 2>/dev/null || true
    fi

    echo "[OK] Import complete."
    echo ""
else
    echo "--- Step 1/3: Import SKIPPED ---"
    echo ""
fi

# ======================== STEP 2: QUANTIZE ========================
if [ "$SKIP_QUANTIZE" -eq 0 ]; then
    echo "--- Step 2/3: Quantize model (${QTYPE}) ---"

    # Determine quantizer from qtype
    case "$QTYPE" in
        uint8)  QUANTIZER="asymmetric_affine" ;;
        int16)  QUANTIZER="dynamic_fixed_point" ;;
        int8|pcq)   QUANTIZER="perchannel_symmetric_affine"; QTYPE="int8" ;;
        bf16)   QUANTIZER="qbfloat16"; QTYPE="qbfloat16" ;;
        *)
            echo "[ERROR] Unknown quantization type: $QTYPE"
            echo "        Supported: uint8, int16, int8/pcq, bf16"
            exit 1
            ;;
    esac

    # Clean up previous quantization
    QFILE="${NAME}_${QTYPE}.quantize"
    rm -f "$QFILE"

    if [ ! -f "$INPUTMETA" ]; then
        echo "[ERROR] Input meta not found: $INPUTMETA"
        echo "        Run export_to_onnx.py with --gen-dataset first."
        exit 1
    fi

    $PEGASUS quantize \
        --model ${NAME}.json \
        --model-data ${NAME}.data \
        --device CPU \
        --with-input-meta ${NAME}_inputmeta.yml \
        --compute-entropy \
        --rebuild \
        --model-quantize ${QFILE} \
        --quantizer ${QUANTIZER} \
        --qtype ${QTYPE}

    if [ ! -f "$QFILE" ]; then
        echo "[ERROR] Quantization failed — ${QFILE} not generated."
        exit 1
    fi

    echo "[OK] Quantization complete: $QFILE"
    echo ""
else
    echo "--- Step 2/3: Quantize SKIPPED ---"
    echo ""
fi

# ======================== STEP 3: EXPORT ========================
echo "--- Step 3/3: Export .nb binary ---"

OUTPUT_DIR="$WORKSPACE/output"
mkdir -p "$OUTPUT_DIR"

if [ "$SKIP_QUANTIZE" -eq 0 ]; then
    QFILE="${NAME}_${QTYPE}.quantize"
    $PEGASUS export ovxlib \
        --model ${NAME}.json \
        --model-data ${NAME}.data \
        --dtype quantized \
        --model-quantize ${QFILE} \
        --target-ide-project 'linux64' \
        --with-input-meta ${NAME}_inputmeta.yml \
        --output-path "${OUTPUT_DIR}/${NAME}_${QTYPE}" \
        --optimize 'VIP8000NANONI_PID0XAD' \
        --pack-nbg-unify \
        --viv-sdk "${VIV_SDK:-/opt/acuity/Vivante_IDE/VivanteIDE5.8.1.1/cmdtools}"
else
    $PEGASUS export ovxlib \
        --model ${NAME}.json \
        --model-data ${NAME}.data \
        --dtype float \
        --target-ide-project 'linux64' \
        --with-input-meta ${NAME}_inputmeta.yml \
        --output-path "${OUTPUT_DIR}/${NAME}_fp16"
fi

# Find the generated .nb file
NB_FILE=$(find "$OUTPUT_DIR" -name "*.nb" -type f 2>/dev/null | head -1)

if [ -n "$NB_FILE" ]; then
    # Copy to a convenient location
    FINAL_NB="$OUTPUT_DIR/yolo26.nb"
    cp "$NB_FILE" "$FINAL_NB"

    echo ""
    echo "======================================================================="
    echo "  Conversion complete!"
    echo "======================================================================="
    echo "  .nb file: $FINAL_NB"
    echo "  Size:     $(du -h "$FINAL_NB" | cut -f1)"
    echo ""
    echo "  Next steps:"
    echo "    1. Copy yolo26.nb to the SD card under NN_MDL/"
    echo "    2. Or include it in amebapro2_fwfs_nn_models.json"
    echo "    3. Flash the firmware and verify detection"
    echo "======================================================================="
else
    echo ""
    echo "[WARN] No .nb file found in output. Check the export logs above."
    echo "       Output directory: $OUTPUT_DIR"
    ls -la "$OUTPUT_DIR/" 2>/dev/null || true
fi
