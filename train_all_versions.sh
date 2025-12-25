#!/bin/bash

# Train YOLOE theo họ num-version (mặc định v8) và biến thể
# Mặc định chỉ train tới m (n, s, m); có thể mở rộng bằng --max hoặc --variants
# Có thể train "hết tất cả" bằng --all (n,s,m,l,x) và --families 8,11
# Ví dụ:
#   ./train_all_versions.sh                     # v8, variants n,s,m
#   ./train_all_versions.sh --family 8 --max s  # v8, variants n,s
#   ./train_all_versions.sh --family 11         # v11 (nếu YAML/weights tồn tại), variants n,s,m
#   ./train_all_versions.sh --variants n,m      # dùng danh sách tùy chọn

cd "$(dirname "$0")" || exit

# Config chung
EPOCHS=300
BATCH=4
IMGSZ=384
LR0=5e-4
WEIGHT_DECAY=0.0005
DEVICE=0
WORKERS=2
PATIENCE=30

# Tham số dòng lệnh
FAMILY=8            # num-version, ví dụ 8, 9, 11...
FAMILIES_CSV=""     # nếu cung cấp, train nhiều họ cùng lúc (vd: 8,11)
MAX_VARIANT="m"     # giới hạn cao nhất của biến thể (n|s|m|l|x)
VARIANTS_CSV=""     # nếu cung cấp, override MAX_VARIANT
DRY_RUN=0
ALL=0               # nếu bật, train n,s,m,l,x

usage() {
    echo "Usage: $0 [--family <num>] [--families <csv>] [--max <n|s|m|l|x>] [--variants <csv>] [--all] [--dry-run]";
    echo "  --family    : Họ num-version của YOLOE (mặc định 8)";
    echo "  --families  : Danh sách họ, vd: 8,11 (override --family)";
    echo "  --max       : Giới hạn biến thể tối đa (mặc định m: n,s,m)";
    echo "  --variants  : Danh sách biến thể tùy chỉnh, vd: n,s,m";
    echo "  --all       : Train tất cả biến thể n,s,m,l,x";
    echo "  --dry-run   : Chỉ in lệnh, không chạy";
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --family)
            FAMILY="$2"; shift 2;;
        --max)
            MAX_VARIANT="$2"; shift 2;;
        --variants)
            VARIANTS_CSV="$2"; shift 2;;
        --families)
            FAMILIES_CSV="$2"; shift 2;;
        --all)
            ALL=1; shift;;
        --dry-run)
            DRY_RUN=1; shift;;
        -h|--help)
            usage; exit 0;;
        *)
            echo "⚠️  Unknown option: $1"; usage; exit 1;;
    esac
done

# Xây dựng danh sách biến thể cần train
declare -a VERSIONS
if [[ -n "$VARIANTS_CSV" ]]; then
    IFS=',' read -r -a VERSIONS <<< "$VARIANTS_CSV"
else
    # Nếu --all, train đầy đủ n,s,m,l,x
    if [[ $ALL -eq 1 ]]; then
        VERSIONS=("n" "s" "m" "l" "x")
    else
        case "$MAX_VARIANT" in
        n) VERSIONS=("n") ;;
        s) VERSIONS=("n" "s") ;;
        m) VERSIONS=("n" "s" "m") ;;
        l) VERSIONS=("n" "s" "m" "l") ;;
        x) VERSIONS=("n" "s" "m" "l" "x") ;;
        *) echo "❌ Invalid --max: $MAX_VARIANT"; exit 1;;
        esac
    fi
fi

declare -a FAMILIES
if [[ -n "$FAMILIES_CSV" ]]; then
    IFS=',' read -r -a FAMILIES <<< "$FAMILIES_CSV"
else
    FAMILIES=("$FAMILY")
fi

echo "🚀 Training YOLOE các họ: ${FAMILIES[*]} với ${#VERSIONS[@]} biến thể: ${VERSIONS[*]}"
echo "Config: batch=$BATCH, imgsz=$IMGSZ, epochs=$EPOCHS, patience=$PATIENCE"
echo "=========================================="
echo ""

# Tạo thư mục pretrain nếu chưa có
mkdir -p pretrain

# Chọn YAML theo họ (family)
resolve_model_yaml() {
    local fam="$1"
    case "$fam" in
        8) echo "ultralytics/cfg/models/v8/yoloe-v8.yaml" ;;
        11) echo "ultralytics/cfg/models/11/yoloe-11.yaml" ;;
        *) echo "" ;;
    esac
}

# Tải weights tự động nếu thiếu (hỗ trợ family=8)
download_weight() {
    local fam="$1"; local var="$2"; local out_path="$3"
    local url=""
    if [[ "$fam" == "8" ]]; then
        case "$var" in
            n) url="https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n.pt" ;;
            s) url="https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8s.pt" ;;
            m) url="https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8m.pt" ;;
            l) url="https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8l.pt" ;;
            x) url="https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8x.pt" ;;
            *) url="" ;;
        esac
    elif [[ "$fam" == "11" ]]; then
        # Nếu có sẵn yolo11n.pt ở repo, tạo symlink cho biến thể n
        if [[ "$var" == "n" && -f "yolo11n.pt" ]]; then
            ln -sf "$(pwd)/yolo11n.pt" "$out_path"
            echo "🔗 Linked local yolo11n.pt → $out_path"
            return 0
        fi
        # Chưa cấu hình URL auto-download cho v11 các biến thể khác
        echo "⚠️  Chưa cấu hình auto-download cho YOLOE v11-$var. Vui lòng đặt weights thủ công."
        return 1
    fi

    if [[ -z "$url" ]]; then
        echo "⚠️  Không có URL download cho v${fam}-${var}"
        return 1
    fi

    echo "🌐 Downloading weights: $url → $out_path"
    if command -v wget >/dev/null 2>&1; then
        wget -q "$url" -O "$out_path" || return 1
    elif command -v curl >/dev/null 2>&1; then
        curl -L "$url" -o "$out_path" || return 1
    else
        echo "❌ Cần 'wget' hoặc 'curl' để tải weights tự động."
        return 1
    fi
    return 0
}

for fam in "${FAMILIES[@]}"; do
    MODEL_BASE_YAML="$(resolve_model_yaml "$fam")"
    if [ -z "$MODEL_BASE_YAML" ] || [ ! -f "$MODEL_BASE_YAML" ]; then
        echo "⚠️  Bỏ qua họ v${fam}: không tìm thấy YAML base ($MODEL_BASE_YAML)"
        continue
    fi

    for VERSION in "${VERSIONS[@]}"; do
        echo ""
        echo "=============================================="
        echo "📊 Training YOLOE v${fam}-${VERSION}"
        echo "=============================================="
    
        # Model config và weights
        MODEL_YAML="$MODEL_BASE_YAML"
        WEIGHTS="yoloe-v${fam}${VERSION}.pt"
        EXP_NAME="fewshot-pe_${fam}${VERSION}_300epochs"
        SAVE_PE="fewshot-pe-${fam}${VERSION}.pt"
    
    # Chọn/Tải weights
    WEIGHTS_PATH="pretrain/$WEIGHTS"
    if [ ! -f "$WEIGHTS_PATH" ]; then
        # Nếu tồn tại file cùng tên ở current dir, dùng nó
        if [ -f "$WEIGHTS" ]; then
            ln -sf "$(pwd)/$WEIGHTS" "$WEIGHTS_PATH"
            echo "🔗 Linked local $WEIGHTS → $WEIGHTS_PATH"
        else
            download_weight "$fam" "$VERSION" "$WEIGHTS_PATH"
            if [ $? -ne 0 ]; then
                echo "   Bỏ qua version này..."
                continue
            fi
        fi
    fi
    
    # Kiểm tra model YAML tồn tại
    if [ -z "$MODEL_YAML" ] || [ ! -f "$MODEL_YAML" ]; then
        echo "⚠️  Model YAML không tồn tại: $MODEL_YAML"
        echo "   Vui lòng đảm bảo có file cấu hình cho YOLOE v${FAMILY}-${VERSION}"
        echo "   Bỏ qua version này..."
        continue
    fi

    echo "Model: $MODEL_YAML"
    echo "Weights: $WEIGHTS_PATH"
    echo "Output: $EXP_NAME"
    echo ""
    
    # Train
    CMD=(python train_fewshot_pe.py
        --data ultralytics/cfg/datasets/fewshot.yaml
        --model "$MODEL_YAML"
        --weights "$WEIGHTS_PATH"
        --epochs $EPOCHS
        --batch $BATCH
        --imgsz $IMGSZ
        --lr0 $LR0
        --weight_decay $WEIGHT_DECAY
        --device $DEVICE
        --workers $WORKERS
        --exp_name "$EXP_NAME"
        --save_pe "$SAVE_PE"
        --patience $PATIENCE)

    if [ $DRY_RUN -eq 1 ]; then
        echo "🔎 Dry-run: ${CMD[*]}"
        EXIT_CODE=0
    else
        "${CMD[@]}"
        EXIT_CODE=$?
    fi
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo ""
        echo "✅ YOLOE v${fam}-${VERSION} training completed!"
        echo "   Best checkpoint: runs/detect/${EXP_NAME}/weights/best.pt"
        echo "   PE file: ${SAVE_PE}"
    else
        echo ""
        echo "❌ YOLOE v${fam}-${VERSION} training failed with exit code $EXIT_CODE"
    fi
    
    echo ""
    echo "=============================================="
    echo ""
    
        # Ngắt giữa các runs (optional)
        sleep 5
    done
done

echo ""
echo "🎉 Hoàn thành training các biến thể đã chọn!"
echo ""
echo "📋 Kết quả:"
for fam in "${FAMILIES[@]}"; do
  for VERSION in "${VERSIONS[@]}"; do
    EXP_NAME="fewshot-pe_${fam}${VERSION}_300epochs"
    BEST_PT="runs/detect/${EXP_NAME}/weights/best.pt"
    PE_FILE="fewshot-pe-${fam}${VERSION}.pt"
    if [ -f "$BEST_PT" ]; then
        echo "   ✅ YOLOE v${fam}-${VERSION}: $BEST_PT"
    else
        echo "   ❌ YOLOE v${fam}-${VERSION}: Not found"
    fi
  done
done
