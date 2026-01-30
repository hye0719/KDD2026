#!/bin/bash

# ==========================================
# Metalens FNO Project - Run Script
# ==========================================
# Usage:
#   ./run.sh train fno 1d
#   ./run.sh train unet 2d
#   ./run.sh train specboost 2d
#   ./run.sh eval fno 1d checkpoints/xxx/fno_1d_best.pth
#   ./run.sh viz fno 1d checkpoints/xxx/fno_1d_best.pth
# ==========================================

set -e  # Exit on error

CONFIG="config/configure.yaml"
COMMAND=$1
MODEL=$2
DIM=$3
CHECKPOINT=$4

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 도움말 출력
show_help() {
    echo ""
    echo -e "${BLUE}==========================================${NC}"
    echo -e "${GREEN}Metalens FNO Project - Run Script${NC}"
    echo -e "${BLUE}==========================================${NC}"
    echo ""
    echo "Usage:"
    echo "  ./run.sh <command> <model> <dim> [checkpoint]"
    echo ""
    echo "Commands:"
    echo "  train   - Train a model"
    echo "  eval    - Evaluate a model (requires checkpoint)"
    echo "  viz     - Visualize and save Top-K results (requires checkpoint)"
    echo ""
    echo "Models:"
    echo "  fno       - Robust FNO model"
    echo "  linear    - Linear MLP model"
    echo "  unet      - U-Net model"
    echo "  specboost - SpecBoost (Two-stage FNO)"
    echo ""
    echo "Dimensions:"
    echo "  1d - 1D configuration"
    echo "  2d - 2D configuration"
    echo ""
    echo "Examples:"
    echo "  ./run.sh train fno 1d"
    echo "  ./run.sh train specboost 2d"
    echo "  ./run.sh eval fno 1d checkpoints/fno_1d_xxx/fno_1d_best.pth"
    echo "  ./run.sh viz unet 2d checkpoints/unet_2d_xxx/unet_2d_best.pth"
    echo ""
    echo "Device Configuration:"
    echo "  Edit config/configure.yaml to change device:"
    echo "    device: 'auto'  # Auto-detect (CUDA > MPS > CPU)"
    echo "    device: 'cuda'  # Force CUDA (NVIDIA GPU)"
    echo "    device: 'mps'   # Force MPS (Apple Silicon)"
    echo "    device: 'cpu'   # Force CPU"
    echo ""
    echo "Data Configuration:"
    echo "  Edit config/configure.yaml to limit data:"
    echo "    max_samples: null   # Use all data"
    echo "    max_samples: 1000   # Use only 1000 samples"
    echo ""
}

# 입력 검증
if [ -z "$COMMAND" ]; then
    show_help
    exit 0
fi

# 기본 검증
if [ -z "$MODEL" ] || [ -z "$DIM" ]; then
    echo -e "${RED}❌ Error: Model and dimension are required${NC}"
    show_help
    exit 1
fi

# 모델 검증
if [[ ! "$MODEL" =~ ^(fno|linear|unet|specboost)$ ]]; then
    echo -e "${RED}❌ Error: Invalid model type: $MODEL${NC}"
    echo "Valid models: fno, linear, unet, specboost"
    exit 1
fi

# 차원 검증
if [[ ! "$DIM" =~ ^(1d|2d)$ ]]; then
    echo -e "${RED}❌ Error: Invalid dimension: $DIM${NC}"
    echo "Valid dimensions: 1d, 2d"
    exit 1
fi

# 명령 실행
case $COMMAND in
    train)
        echo ""
        echo -e "${GREEN}==========================================${NC}"
        echo -e "${GREEN}🚀 Training ${MODEL^^} (${DIM^^})${NC}"
        echo -e "${GREEN}==========================================${NC}"
        echo ""
        
        if [ "$MODEL" == "specboost" ]; then
            python train_specboost.py --config $CONFIG --dim $DIM
        else
            python train.py --config $CONFIG --model $MODEL --dim $DIM
        fi
        
        echo ""
        echo -e "${GREEN}✅ Training complete!${NC}"
        echo -e "${YELLOW}📁 Check ./checkpoints for saved models${NC}"
        echo -e "${YELLOW}📁 Check ./results for visualization${NC}"
        ;;
    
    eval)
        if [ -z "$CHECKPOINT" ]; then
            echo -e "${RED}❌ Error: Checkpoint path is required for eval command${NC}"
            exit 1
        fi
        
        echo ""
        echo -e "${GREEN}==========================================${NC}"
        echo -e "${GREEN}📊 Evaluating ${MODEL^^} (${DIM^^})${NC}"
        echo -e "${GREEN}==========================================${NC}"
        echo ""
        
        python evaluation.py --config $CONFIG --checkpoint $CHECKPOINT --model $MODEL --dim $DIM
        ;;
    
    viz)
        if [ -z "$CHECKPOINT" ]; then
            echo -e "${RED}❌ Error: Checkpoint path is required for viz command${NC}"
            exit 1
        fi
        
        echo ""
        echo -e "${GREEN}==========================================${NC}"
        echo -e "${GREEN}📈 Visualizing ${MODEL^^} (${DIM^^})${NC}"
        echo -e "${GREEN}==========================================${NC}"
        echo ""
        
        python main.py viz --config $CONFIG --checkpoint $CHECKPOINT --model $MODEL --dim $DIM
        ;;
    
    *)
        echo -e "${RED}❌ Error: Unknown command: $COMMAND${NC}"
        show_help
        exit 1
        ;;
esac
