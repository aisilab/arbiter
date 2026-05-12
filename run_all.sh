#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# run_all.sh — End-to-end Arbiter experiment pipeline
# ---------------------------------------------------------------------------
# Steps: install → generate_conversations → run_experiments → analyze_experiments
#
# Usage:
#   ./run_all.sh                    # run everything, 10 conversation variants
#   ./run_all.sh -n 5               # run everything, 5 conversation variants
#   ./run_all.sh --skip-install     # skip package installation
#   ./run_all.sh --dry-run          # dry-run experiments (no API calls)
#   ./run_all.sh -n 3 --reps 5      # 3 conv variants, 5 replications per cell
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
SKIP_ANALYSIS=false
OUTPUT_BASE="results/v0.6"

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
    --output-dir)
      OUTPUT_BASE="$2"
      shift 2
      ;;
    -h|--help)
      echo "Usage: $0 [options]"
      echo ""
      echo "Options:"
      echo "  -n, --num-variants N   Number of conversation variants per config (default: 10)"
      echo "  --reps N               Number of replications per experiment cell (default: 10)"
      echo "  --skip-install         Skip pip install -e ."
      echo "  --dry-run              Dry-run mode (print what would run)"
      echo "  --skip-conversations   Skip conversation generation"
      echo "  --skip-experiments     Skip experiment runs"
      echo "  --skip-analysis        Skip analysis"
      echo "  --output-dir PATH      Base output directory (default: results/v0.6)"
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
echo "  variants : $NUM_VARIANTS"
echo "  reps     : $REPS"
echo "  output   : $OUTPUT_BASE"
echo "  dry-run  : ${DRY_RUN:-(none)}"
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
      --output-dir "$OUTPUT_BASE" \
      --skip-existing \
      -n "$NUM_VARIANTS"
else
  echo "[2/4] Skipping conversation generation (--skip-conversations)"
fi

echo ""

# ---------------------------------------------------------------------------
# 3. Run experiments
# ---------------------------------------------------------------------------
if [[ "$SKIP_EXPERIMENTS" == false ]]; then
  echo "[3/4] Running experiments (reps=$REPS)..."
  python3 run_experiment.py \
      --replications "$REPS" \
      $DRY_RUN
else
  echo "[3/4] Skipping experiment runs (--skip-experiments)"
fi

echo ""

# ---------------------------------------------------------------------------
# 4. Analyze experiments
# ---------------------------------------------------------------------------
if [[ "$SKIP_ANALYSIS" == false ]]; then
  echo "[4/4] Analyzing experiments..."
  python3 analyze_experiments.py "$OUTPUT_BASE"
else
  echo "[4/4] Skipping analysis (--skip-analysis)"
fi

echo ""
echo "========================================"
echo "  PIPELINE COMPLETE"
echo "========================================"
echo "  Results: $OUTPUT_BASE/"
echo "  Stats  : $(dirname "$OUTPUT_BASE")/analysis_stats.json"
echo "  Table  : $(dirname "$OUTPUT_BASE")/analysis_stats.md"
