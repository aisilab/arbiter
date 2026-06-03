#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# run_all.sh — End-to-end Arbiter experiment pipeline
# ---------------------------------------------------------------------------
# Steps: install → generate_conversations → run_api_experiments →
#        run_offline_experiments → analyze_all
#
# Usage:
#   ./run_all.sh                           # run everything
#   ./run_all.sh -n 5                      # run everything, 5 conversation variants
#   ./run_all.sh --skip-install            # skip package installation
#   ./run_all.sh --skip-offline            # skip offline judge experiments
#   ./run_all.sh --dry-run                 # dry-run mode (no API calls)
#   ./run_all.sh -n 3 --reps 5             # 3 conv variants, 5 replications per cell
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Defaults
NUM_VARIANTS=5
REPS=20
SKIP_INSTALL=false
DRY_RUN=""
SKIP_CONVERSATIONS=false
SKIP_EXPERIMENTS=false
SKIP_OFFLINE=false
SKIP_API=false
SKIP_ANALYSIS=false
V07_BASE="results/v0.7"
CONVERSATIONS_DIR="$V07_BASE/conversations"
ARBITER_API_DIR="$V07_BASE/arbiter/api"
ARBITER_OFFLINE_DIR="$V07_BASE/arbiter/offline"

# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case $1 in
    -n|--num-variants)
      NUM_VARIANTS="$2"
      shift 2
      ;;
    --reps)
      REPS="$2"
      shift 2
      ;;
    --skip-install)
      SKIP_INSTALL=true
      shift
      ;;
    --dry-run)
      DRY_RUN="--dry-run"
      shift
      ;;
    --skip-conversations)
      SKIP_CONVERSATIONS=true
      shift
      ;;
    --skip-experiments)
      SKIP_EXPERIMENTS=true
      shift
      ;;
    --skip-analysis)
      SKIP_ANALYSIS=true
      shift
      ;;
    --skip-offline)
      SKIP_OFFLINE=true
      shift
      ;;
    --skip-api)
      SKIP_API=true
      shift
      ;;
    --output-dir)
      V07_BASE="$2"
      CONVERSATIONS_DIR="$V07_BASE/conversations"
      ARBITER_API_DIR="$V07_BASE/arbiter/api"
      ARBITER_OFFLINE_DIR="$V07_BASE/arbiter/offline"
      shift 2
      ;;
    -h|--help)
      echo "Usage: $0 [options]"
      echo ""
      echo "Options:"
      echo "  -n, --num-variants N   Number of conversation variants per config (default: 5)"
      echo "  --reps N               Number of replications per experiment cell (default: 5)"
      echo "  --skip-install         Skip pip install -e ."
      echo "  --dry-run              Dry-run mode (print what would run)"
      echo "  --skip-conversations   Skip conversation generation"
      echo "  --skip-experiments     Skip experiment runs"
  echo "  --skip-offline         Skip offline judge experiments"
  echo "  --skip-api            Skip API judge experiments"
  echo "  --skip-analysis        Skip analysis"
      echo "  --output-dir PATH      Base output directory (default: results/v0.7)"
      echo "  -h, --help             Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Run '$0 --help' for usage."
      exit 1
      ;;
  esac
done

# ---------------------------------------------------------------------------
# 1. Install package
# ---------------------------------------------------------------------------
echo "========================================"
echo "  ARBITER EXPERIMENT PIPELINE"
echo "========================================"
echo ""
echo "  variants             : $NUM_VARIANTS"
echo "  reps                 : $REPS"
echo "  base                 : $V07_BASE"
echo "  conversations        : $CONVERSATIONS_DIR"
echo "  arbiter API out      : $ARBITER_API_DIR"
echo "  arbiter offline out  : $ARBITER_OFFLINE_DIR"
  echo "  skip-offline         : $SKIP_OFFLINE"
  echo "  skip-api             : $SKIP_API"
  echo "  dry-run              : ${DRY_RUN:-(none)}"
echo ""

if [[ "$SKIP_INSTALL" == false ]]; then
  echo "[1/4] Installing arbiter package..."
  pip install -e .
  pip install ag2
else
  echo "[1/4] Skipping installation (--skip-install)"
fi

echo ""

# ---------------------------------------------------------------------------
# 2. Generate conversations
# ---------------------------------------------------------------------------
if [[ "$SKIP_CONVERSATIONS" == false ]]; then
  echo "[2/4] Generating conversations (n=$NUM_VARIANTS)..."
  python3 generate_conversations.py \
      --output-dir "$CONVERSATIONS_DIR" \
      --skip-existing \
      -n "$NUM_VARIANTS"
else
  echo "[2/4] Skipping conversation generation (--skip-conversations)"
fi

echo ""

# ---------------------------------------------------------------------------
# 3. Run experiments (API backend)
# ---------------------------------------------------------------------------
if [[ "$SKIP_EXPERIMENTS" == false && "$SKIP_API" == false ]]; then
  echo "[3/6] Running API experiments (reps=$REPS)..."
  python3 run_experiment.py \
      --replications "$REPS" \
      --backend api \
      $DRY_RUN
elif [[ "$SKIP_API" == true ]]; then
  echo "[3/6] Skipping API experiments (--skip-api)"
else
  echo "[3/6] Skipping experiment runs (--skip-experiments)"
fi

echo ""

# ---------------------------------------------------------------------------
# 4. Run experiments (offline backend)
# ---------------------------------------------------------------------------
if [[ "$SKIP_EXPERIMENTS" == false && "$SKIP_OFFLINE" == false ]]; then
  echo "[4/6] Running offline experiments (reps=$REPS)..."
  python3 run_experiment.py \
      --replications "$REPS" \
      --backend offline \
      $DRY_RUN
elif [[ "$SKIP_EXPERIMENTS" == false && "$SKIP_OFFLINE" == true ]]; then
  echo "[4/6] Skipping offline experiments (--skip-offline)"
fi

echo ""

# ---------------------------------------------------------------------------
# 5. Analyze API experiments
# ---------------------------------------------------------------------------
if [[ "$SKIP_ANALYSIS" == false && "$SKIP_API" == false ]]; then
  echo "[5/6] Analyzing API experiments..."
  python3 analyze_experiments.py "$ARBITER_API_DIR" \
      --output "$V07_BASE/analysis_stats_api.json"
elif [[ "$SKIP_ANALYSIS" == false && "$SKIP_API" == true ]]; then
  echo "[5/6] Skipping API analysis (--skip-api)"
else
  echo "[5/6] Skipping analysis (--skip-analysis)"
fi

echo ""

# ---------------------------------------------------------------------------
# 6. Analyze offline experiments
# ---------------------------------------------------------------------------
if [[ "$SKIP_ANALYSIS" == false && "$SKIP_OFFLINE" == false ]]; then
  echo "[6/6] Analyzing offline experiments..."
  python3 analyze_experiments.py "$ARBITER_OFFLINE_DIR" \
      --output "$V07_BASE/analysis_stats_offline.json"
elif [[ "$SKIP_ANALYSIS" == false && "$SKIP_OFFLINE" == true ]]; then
  echo "[6/6] Skipping offline analysis (--skip-offline)"
fi

echo ""
echo "========================================"
echo "  PIPELINE COMPLETE"
echo "========================================"
echo "  Conversations        : $CONVERSATIONS_DIR/"
echo "  Arbiter API out      : $ARBITER_API_DIR/"
echo "  Arbiter offline out  : $ARBITER_OFFLINE_DIR/"
echo "  Stats (API)          : $V07_BASE/analysis_stats_api.json"
echo "  Stats (offline)      : $V07_BASE/analysis_stats_offline.json"
