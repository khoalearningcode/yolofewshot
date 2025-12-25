#!/bin/bash

# Train YOLOE tất cả họ (v8, v11, v10, v9, v3, v5, v6) với tất cả biến thể (n,s,m,l,x)
# Chạy một phát hết tất cả luôn

cd "$(dirname "$0")" || exit

# Danh sách tất cả họ có sẵn YAML
FAMILIES=(8 11 10 9 3 5 6)

# Danh sách biến thể
VARIANTS="n s m"

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  🚀 Training YOLOE - Tất cả họ × n,s,m                         ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "Họ: ${FAMILIES[*]}"
echo "Biến thể: $VARIANTS"
echo ""
echo "⚠️  Cảnh báo:"
echo "  - Sẽ train 7 họ × 3 biến thể = 21 model"
echo "  - Có thể mất vài ngày hoặc vài tuần"
echo "  - Để máy chạy qua đêm/weekend"
echo "  - Nếu weights thiếu, script sẽ tự tải (chỉ v8 có auto-download)"
echo ""
read -p "Tiếp tục? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Đã hủy."
    exit 0
fi

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  Bắt đầu training...                                           ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Theo dõi thống kê
TOTAL=0
SUCCESS=0
FAILED=0
SKIPPED=0

# Lặp qua từng họ
for FAMILY in "${FAMILIES[@]}"; do
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📦 Họ: YOLOE v${FAMILY}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # Kiểm tra YAML base có tồn tại không
    YAML_FOUND=0
    for YAML_PATTERN in "ultralytics/cfg/models/v${FAMILY}/yoloe-v${FAMILY}.yaml" \
                         "ultralytics/cfg/models/v${FAMILY}/yolo${FAMILY}.yaml" \
                         "ultralytics/cfg/models/${FAMILY}/yoloe-${FAMILY}.yaml" \
                         "ultralytics/cfg/models/${FAMILY}/yolo${FAMILY}.yaml"; do
        if [ -f "$YAML_PATTERN" ]; then
            YAML_FOUND=1
            echo "✅ Tìm YAML: $YAML_PATTERN"
            break
        fi
    done
    
    if [ $YAML_FOUND -eq 0 ]; then
        echo "⚠️  Không tìm YAML base cho v${FAMILY}, bỏ qua họ này"
        SKIPPED=$((SKIPPED + 3))
        TOTAL=$((TOTAL + 3))
        continue
    fi
    
    # Train từng biến thể
    for VAR in $VARIANTS; do
        TOTAL=$((TOTAL + 1))
        
        echo ""
        echo "  📊 Training v${FAMILY}-${VAR}..."
        
        FAMILY=$FAMILY bash train_simple.sh "$VAR" 2>&1
        EXIT_CODE=$?
        
        if [ $EXIT_CODE -eq 0 ]; then
            SUCCESS=$((SUCCESS + 1))
            echo "  ✅ v${FAMILY}-${VAR} OK"
        else
            FAILED=$((FAILED + 1))
            echo "  ❌ v${FAMILY}-${VAR} FAILED"
        fi
    done
done

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  📊 Kết quả cuối cùng                                          ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "  Total: $TOTAL models"
echo "  ✅ Thành công: $SUCCESS"
echo "  ❌ Thất bại: $FAILED"
echo "  ⏭️  Bỏ qua: $SKIPPED"
echo ""

if [ $FAILED -eq 0 ]; then
    echo "🎉 Tất cả training xong!"
else
    echo "⚠️  Có $FAILED model thất bại, vui lòng kiểm tra logs"
fi

echo ""
