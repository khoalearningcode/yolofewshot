#!/bin/bash

# Train YOLOE các biến thể (n,s,m,l,x)
# Giữ nguyên YAML base theo họ (family)
# Chỉ thay weights theo biến thể

cd "$(dirname "$0")" || exit

# Config
EPOCHS=300
BATCH=4
IMGSZ=384
LR0=5e-4
WEIGHT_DECAY=0.0005
DEVICE=0
WORKERS=2
PATIENCE=30

# Tham số
FAMILY="${FAMILY:-8}"  # Họ (8, 11, 10, 9, 3, 5, 6...)
VARIANTS="${1:-m}"    # Biến thể (mặc định m)

# Kiểm tra YAML base theo họ
case "$FAMILY" in
  8) MODEL_DIR="ultralytics/cfg/models/v8" ;;
  11) MODEL_DIR="ultralytics/cfg/models/11" ;;
  10) MODEL_DIR="ultralytics/cfg/models/v10" ;;
  9) MODEL_DIR="ultralytics/cfg/models/v9" ;;
  3) MODEL_DIR="ultralytics/cfg/models/v3" ;;
  5) MODEL_DIR="ultralytics/cfg/models/v5" ;;
  6) MODEL_DIR="ultralytics/cfg/models/v6" ;;
  *) echo "❌ Họ v${FAMILY} không hỗ trợ"; exit 1 ;;
esac

# Tìm YAML base (yoloe-v{family}.yaml hoặc yolo{family}.yaml)
if [ -f "$MODEL_DIR/yoloe-v${FAMILY}.yaml" ]; then
    MODEL_YAML="$MODEL_DIR/yoloe-v${FAMILY}.yaml"
elif [ -f "$MODEL_DIR/yolo${FAMILY}.yaml" ]; then
    MODEL_YAML="$MODEL_DIR/yolo${FAMILY}.yaml"
else
    echo "❌ Không tìm YAML base cho v${FAMILY} tại $MODEL_DIR"
    exit 1
fi

echo "🚀 Training YOLOE v${FAMILY} với YAML: $MODEL_YAML"
echo "Biến thể: $VARIANTS"
echo "Config: batch=$BATCH, imgsz=$IMGSZ, epochs=$EPOCHS, patience=$PATIENCE"
echo "=========================================="
echo ""

# Chuyển đổi dạng biến thể (từ "n s m" hoặc "n,s,m" thành array)
VARIANTS=$(echo "$VARIANTS" | tr ',' ' ')

for VAR in $VARIANTS; do
    echo ""
    echo "=============================================="
    echo "📊 Training YOLOE v${FAMILY}-${VAR}"
    echo "=============================================="
    
    WEIGHTS="yoloe-v${FAMILY}${VAR}.pt"
    EXP_NAME="fewshot-pe_${FAMILY}${VAR}_300epochs"
    SAVE_PE="fewshot-pe-${FAMILY}${VAR}.pt"
    
    # Kiểm tra/tải weights
    if [ ! -f "pretrain/$WEIGHTS" ] && [ ! -f "$WEIGHTS" ]; then
        echo "⚠️  Weights không tồn tại: $WEIGHTS"
        mkdir -p pretrain
        
        # Chỉ auto-download cho v8
        if [ "$FAMILY" == "8" ]; then
            echo "   Tự động tải..."
            URL=""
            case "$VAR" in
                n) URL="https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n.pt" ;;
                s) URL="https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8s.pt" ;;
                m) URL="https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8m.pt" ;;
                l) URL="https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8l.pt" ;;
                x) URL="https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8x.pt" ;;
            esac
            
            if [ -n "$URL" ]; then
                echo "🌐 Downloading from $URL"
                if command -v wget >/dev/null 2>&1; then
                    wget -q "$URL" -O "pretrain/$WEIGHTS" || {
                        echo "❌ Download failed"; continue
                    }
                elif command -v curl >/dev/null 2>&1; then
                    curl -L "$URL" -o "pretrain/$WEIGHTS" || {
                        echo "❌ Download failed"; continue
                    }
                else
                    echo "❌ Cần wget hoặc curl"; continue
                fi
            fi
        else
            echo "   Để download cho v${FAMILY}, đặt weights thủ công vào pretrain/"
            echo "   Bỏ qua biến thể này"
            continue
        fi
    fi
    
    # Chọn weights path
    if [ -f "pretrain/$WEIGHTS" ]; then
        WEIGHTS_PATH="pretrain/$WEIGHTS"
    else
        WEIGHTS_PATH="$WEIGHTS"
    fi
    
    echo "Model: $MODEL_YAML"
    echo "Weights: $WEIGHTS_PATH"
    echo "Output: $EXP_NAME"
    echo ""
    
    # Train
    python train_fewshot_pe.py \
        --data ultralytics/cfg/datasets/fewshot.yaml \
        --model "$MODEL_YAML" \
        --weights "$WEIGHTS_PATH" \
        --epochs $EPOCHS \
        --batch $BATCH \
        --imgsz $IMGSZ \
        --lr0 $LR0 \
        --weight_decay $WEIGHT_DECAY \
        --device $DEVICE \
        --workers $WORKERS \
        --exp_name "$EXP_NAME" \
        --save_pe "$SAVE_PE" \
        --patience $PATIENCE
    
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo ""
        echo "✅ v${FAMILY}-${VAR} completed!"
        echo "   Checkpoint: runs/detect/${EXP_NAME}/weights/best.pt"
    else
        echo ""
        echo "❌ v${FAMILY}-${VAR} failed (exit code $EXIT_CODE)"
    fi
    
    # Cleanup GPU memory
    echo ""
    echo "🧹 Cleaning GPU memory..."
    python cleanup_gpu.py
    sleep 3
    
    echo ""
    sleep 2
done

echo ""
echo "🎉 Done!"
