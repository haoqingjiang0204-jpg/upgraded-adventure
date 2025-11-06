#!/bin/bash

# =======================================================
# Transformer从零实现 - 自动化运行脚本
# 使用方法: ./scripts/run.sh [mode]
# mode: train | ablation | generate | all
# =======================================================

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 参数设置
MODE=${1:-all}
SEED=42
DEVICE="cuda:0"
CONFIG="configs/base.yaml"
CHECKPOINT="checkpoints/best_model.pth"

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查环境
check_environment() {
    log_info "检查Python环境..."
    if ! command -v python &> /dev/null; then
        log_error "Python未安装"
        exit 1
    fi

    python -c "import torch; print(f'PyTorch版本: {torch.__version__}')"
    python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}')"
    if [ "$DEVICE" != "cpu" ] && ! python -c "import torch; assert torch.cuda.is_available()" &> /dev/null; then
        log_warning "CUDA不可用，将使用CPU"
        DEVICE="cpu"
    fi
}

# 创建目录结构
setup_directories() {
    log_info "创建目录结构..."
    mkdir -p checkpoints results/configs results/plots
}

# 训练模型
train_model() {
    log_info "开始训练模型..."
    log_info "设备: $DEVICE, 随机种子: $SEED"

    python src/main.py \
        --mode train \
        --config $CONFIG \
        --seed $SEED \
        --device $DEVICE

    # 检查训练是否成功
    if [ -f "checkpoints/best_model.pth" ]; then
        log_success "模型训练完成，最佳模型已保存"
    else
        log_error "模型训练失败"
        exit 1
    fi
}

# 运行消融实验
run_ablation_study() {
    log_info "开始消融实验..."

    # 完整Encoder-Decoder
    log_info "1. 训练完整Encoder-Decoder架构..."
    python src/main.py \
        --mode train \
        --config configs/base.yaml \
        --seed $SEED \
        --device $DEVICE

    # Encoder-only
    log_info "2. 训练Encoder-only架构..."
    python src/main.py \
        --mode train \
        --config configs/encoder_only.yaml \
        --seed $SEED \
        --device $DEVICE

    # 无位置编码
    log_info "3. 训练无位置编码变体..."
    python src/main.py \
        --mode train \
        --config configs/ablation/no_positional_encoding.yaml \
        --seed $SEED \
        --device $DEVICE

    # 单头注意力
    log_info "4. 训练单头注意力变体..."
    python src/main.py \
        --mode train \
        --config configs/ablation/single_head.yaml \
        --seed $SEED \
        --device $DEVICE

    # 生成消融实验报告
    log_info "生成消融实验报告..."
    python src/experiments.py \
        --config $CONFIG \
        --seed $SEED \
        --device $DEVICE

    log_success "消融实验完成"
}

# 文本生成测试
generate_text() {
    log_info "测试文本生成..."

    if [ ! -f "$CHECKPOINT" ]; then
        log_warning "未找到训练好的模型，请先运行训练"
        return 1
    fi

    # 测试不同的提示文本
    prompts=("ROMEO:" "KING:" "JULIET:" "Hello world" "The future of")

    for prompt in "${prompts[@]}"; do
        log_info "生成提示: '$prompt'"
        python src/main.py \
            --mode generate \
            --prompt "$prompt" \
            --checkpoint $CHECKPOINT \
            --seed $SEED \
            --temperature 0.8 \
            --max_length 50
        echo "----------------------------------------"
    done
}

# 生成实验图表
generate_plots() {
    log_info "生成实验图表..."

    python src/plot_results.py \
        --results_dir results \
        --output_dir results/plots

    if [ -f "results/plots/training_curves.png" ]; then
        log_success "实验图表已生成"
    else
        log_warning "图表生成可能存在问题"
    fi
}

# 验证实验结果
validate_results() {
    log_info "验证实验结果..."

    python src/validate.py \
        --checkpoint $CHECKPOINT \
        --seed $SEED \
        --device $DEVICE

    log_success "实验结果验证完成"
}

# 主函数
main() {
    log_info "================================================"
    log_info "    Transformer从零实现 - 实验自动化脚本"
    log_info "================================================"

    # 检查环境
    check_environment

    # 创建目录
    setup_directories

    case $MODE in
        "train")
            train_model
            ;;
        "ablation")
            run_ablation_study
            ;;
        "generate")
            generate_text
            ;;
        "all")
            log_info "执行完整实验流程..."
            train_model
            run_ablation_study
            generate_text
            generate_plots
            validate_results
            ;;
        *)
            log_error "未知模式: $MODE"
            log_info "可用模式: train | ablation | generate | all"
            exit 1
            ;;
    esac

    log_success "实验完成！"
    log_info "结果文件:"
    ls -la results/
    log_info "模型文件:"
    ls -la checkpoints/
}

# 显示帮助信息
show_help() {
    echo "使用方法: $0 [mode]"
    echo ""
    echo "模式:"
    echo "  train     训练完整模型"
    echo "  ablation  运行消融实验"
    echo "  generate  文本生成测试"
    echo "  all       执行完整实验流程（默认）"
    echo ""
    echo "环境变量:"
    echo "  SEED      随机种子（默认: 42）"
    echo "  DEVICE    训练设备（默认: cuda:0）"
    echo ""
    echo "示例:"
    echo "  $0 train                    # 训练模型"
    echo "  $0 ablation                 # 运行消融实验"
    echo "  DEVICE=cpu $0 all           # 使用CPU运行完整实验"
    echo "  SEED=123 $0 train           # 使用指定随机种子训练"
}

# 处理帮助参数
if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    show_help
    exit 0
fi

# 读取环境变量
if [ ! -z "$SEED_ENV" ]; then
    SEED=$SEED_ENV
fi

if [ ! -z "$DEVICE_ENV" ]; then
    DEVICE=$DEVICE_ENV
fi

# 执行主函数
main