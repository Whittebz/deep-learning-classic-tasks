#!/bin/bash
# ============================================================================
# setup_envs.sh - 一键创建/管理 Conda 环境
# 用法:
#   bash setup_envs.sh              # 创建所有 10 个环境
#   bash setup_envs.sh --task 1     # 只创建任务 1 的环境
#   bash setup_envs.sh --task 1 --force  # 强制重建任务 1 的环境
#   bash setup_envs.sh --list       # 列出所有已创建的 dl_task 环境
#   bash setup_envs.sh --clean      # 删除所有 dl_task 环境
# ============================================================================

set -e

# 项目根目录（脚本所在目录）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 所有任务目录（按顺序）
TASK_DIRS=(
    "01_image_classification"
    "02_object_detection"
    "03_semantic_segmentation"
    "04_sentiment_analysis"
    "05_machine_translation"
    "06_named_entity_recognition"
    "07_text_summarization"
    "08_speech_recognition"
    "09_image_generation"
    "10_time_series_forecasting"
)

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo ""
    echo -e "${BLUE}============================================${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}============================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# 检查 conda 是否可用
check_conda() {
    if ! command -v conda &> /dev/null; then
        print_error "conda 未找到。请先安装 Miniconda。"
        echo "安装链接: https://docs.conda.io/en/latest/miniconda.html"
        exit 1
    fi
    print_success "conda 已就绪: $(conda --version)"
}

# 检查环境是否存在
env_exists() {
    conda env list | grep -q "^$1 "
}

# 创建单个环境
create_env() {
    local task_idx=$1  # 0-indexed
    local task_dir="${TASK_DIRS[$task_idx]}"
    local env_file="${SCRIPT_DIR}/${task_dir}/environment.yml"
    local env_name="dl_task$(printf '%02d' $((task_idx + 1)))"

    if [ ! -f "$env_file" ]; then
        print_error "环境文件不存在: ${env_file}"
        return 1
    fi

    if [ "$FORCE" = true ]; then
        if env_exists "$env_name"; then
            print_warning "正在删除已有环境: ${env_name}"
            conda env remove -n "$env_name" -y 2>/dev/null || true
        fi
    fi

    if env_exists "$env_name"; then
        echo ""
        echo -e "${BLUE}▶ 环境已存在，正在同步依赖 [${env_name}] 来自 ${task_dir}/environment.yml ...${NC}"
        echo "  这可能需要几分钟，取决于网络速度..."

        if conda env update -n "$env_name" -f "$env_file" --prune 2>&1; then
            print_success "环境同步成功: ${env_name}"
            return 0
        else
            print_error "环境同步失败: ${env_name}"
            return 1
        fi
    fi

    echo ""
    echo -e "${BLUE}▶ 正在创建环境 [${env_name}] 来自 ${task_dir}/environment.yml ...${NC}"
    echo "  这可能需要几分钟，取决于网络速度..."
    
    if conda env create -f "$env_file" 2>&1; then
        print_success "环境创建成功: ${env_name}"
    else
        print_error "环境创建失败: ${env_name}"
        return 1
    fi
}

# 列出所有 dl_task 环境
list_envs() {
    print_header "已创建的 dl_task 环境"
    local found=false
    while IFS= read -r line; do
        if [[ "$line" == dl_task* ]]; then
            echo -e "  ${GREEN}●${NC} $line"
            found=true
        fi
    done < <(conda env list)
    
    if [ "$found" = false ]; then
        print_warning "尚未创建任何 dl_task 环境"
    fi
    echo ""
}

# 清理所有 dl_task 环境
clean_envs() {
    print_header "删除所有 dl_task 环境"
    for i in $(seq 1 10); do
        local env_name="dl_task$(printf '%02d' $i)"
        if env_exists "$env_name"; then
            echo -e "  正在删除: ${env_name}..."
            conda env remove -n "$env_name" -y 2>/dev/null || true
            print_success "已删除: ${env_name}"
        fi
    done
    echo ""
    print_success "清理完成"
}

# ============================================================================
# 参数解析
# ============================================================================
TASK_ID=""
FORCE=false
ACTION="create"

while [[ $# -gt 0 ]]; do
    case $1 in
        --task)
            TASK_ID="$2"
            shift 2
            ;;
        --force)
            FORCE=true
            shift
            ;;
        --list)
            ACTION="list"
            shift
            ;;
        --clean)
            ACTION="clean"
            shift
            ;;
        --help|-h)
            echo "用法: bash setup_envs.sh [选项]"
            echo ""
            echo "选项:"
            echo "  --task N     只创建任务 N 的环境 (N=1-10)"
            echo "  --force      强制重建已有环境"
            echo "  --list       列出所有已创建的 dl_task 环境"
            echo "  --clean      删除所有 dl_task 环境"
            echo "  --help, -h   显示此帮助信息"
            exit 0
            ;;
        *)
            print_error "未知参数: $1"
            echo "使用 --help 查看帮助"
            exit 1
            ;;
    esac
done

# ============================================================================
# 执行
# ============================================================================
check_conda

case $ACTION in
    list)
        list_envs
        ;;
    clean)
        clean_envs
        ;;
    create)
        if [ -n "$TASK_ID" ]; then
            # 创建单个任务的环境
            if [[ "$TASK_ID" -lt 1 || "$TASK_ID" -gt 10 ]]; then
                print_error "任务编号必须在 1-10 之间，收到: ${TASK_ID}"
                exit 1
            fi
            task_idx=$((TASK_ID - 1))
            print_header "创建任务 ${TASK_ID} 的 Conda 环境"
            create_env $task_idx
        else
            # 创建所有环境
            print_header "创建全部 10 个 Conda 环境"
            echo ""
            echo "注意: 首次创建全部环境可能需要 30-60 分钟（取决于网络速度）"
            echo "建议使用 --task N 逐个创建，便于排查问题"
            echo ""
            
            success_count=0
            fail_count=0
            for i in $(seq 0 9); do
                if create_env $i; then
                    ((success_count++))
                else
                    ((fail_count++))
                fi
            done
            
            echo ""
            print_header "创建完成"
            print_success "成功: ${success_count} 个环境"
            if [ $fail_count -gt 0 ]; then
                print_error "失败: ${fail_count} 个环境"
            fi
        fi
        ;;
esac

echo ""
echo "提示: 使用 'bash setup_envs.sh --list' 查看已创建的环境"
echo "提示: 使用 'python start_train.py --task N' 训练时会自动激活对应环境"
