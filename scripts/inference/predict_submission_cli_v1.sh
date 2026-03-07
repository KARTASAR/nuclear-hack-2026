#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
PREDICT_SCRIPT="${ROOT_DIR}/scripts/inference/predict_submission_v1.py"

MANIFEST="${ROOT_DIR}/artifacts/submission_pack_v1/manifest.json"
REGION=""
CENTER=""
INPUT_TXT=""
SINGLE_POLICY="fallback"
OUTPUT_JSON=""
DEFAULT_OUTPUT_DIR="${ROOT_DIR}/artifacts/predictions"

usage() {
  cat <<EOF
Usage:
  scripts/inference/predict_submission_cli_v1.sh \\
    --region <cortex|striatum|cerebellum> \\
    --center <1500|2900> \\
    --input <path/to/spectrum.txt> \\
    [--manifest <path/to/manifest.json>] \\
    [--single-policy <fallback|main>] \\
    [--output-json <path/to/output.json>]
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --manifest)
      MANIFEST="$2"
      shift 2
      ;;
    --region)
      REGION="$2"
      shift 2
      ;;
    --center)
      CENTER="$2"
      shift 2
      ;;
    --input)
      INPUT_TXT="$2"
      shift 2
      ;;
    --single-policy)
      SINGLE_POLICY="$2"
      shift 2
      ;;
    --output-json)
      OUTPUT_JSON="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ -z "${REGION}" || -z "${CENTER}" || -z "${INPUT_TXT}" ]]; then
  echo "Error: --region, --center, --input are required." >&2
  usage
  exit 2
fi

if [[ "${CENTER}" != "1500" && "${CENTER}" != "2900" ]]; then
  echo "Error: --center must be 1500 or 2900." >&2
  exit 2
fi

if [[ "${REGION}" != "cortex" && "${REGION}" != "striatum" && "${REGION}" != "cerebellum" ]]; then
  echo "Error: --region must be one of: cortex, striatum, cerebellum." >&2
  exit 2
fi

if [[ "${SINGLE_POLICY}" != "fallback" && "${SINGLE_POLICY}" != "main" ]]; then
  echo "Error: --single-policy must be fallback or main." >&2
  exit 2
fi

if [[ ! -f "${PYTHON_BIN}" ]]; then
  echo "Error: Python not found: ${PYTHON_BIN}" >&2
  exit 2
fi

if [[ ! -f "${PREDICT_SCRIPT}" ]]; then
  echo "Error: predict script not found: ${PREDICT_SCRIPT}" >&2
  exit 2
fi

if [[ ! -f "${MANIFEST}" ]]; then
  echo "Error: manifest not found: ${MANIFEST}" >&2
  echo "Build it first: .venv/bin/python scripts/inference/build_submission_pack_v1.py --out-dir artifacts/submission_pack_v1" >&2
  exit 2
fi

if [[ ! -f "${INPUT_TXT}" ]]; then
  echo "Error: input spectrum file not found: ${INPUT_TXT}" >&2
  exit 2
fi

OUT_FP="${OUTPUT_JSON}"
if [[ -z "${OUT_FP}" ]]; then
  TS="$(date +%Y%m%d_%H%M%S)"
  RUN_DIR="${DEFAULT_OUTPUT_DIR}/${TS}_single_center_${REGION}_${CENTER}_${SINGLE_POLICY}"
  mkdir -p "${RUN_DIR}"
  OUT_FP="${RUN_DIR}/prediction.json"
fi

"${PYTHON_BIN}" "${PREDICT_SCRIPT}" \
  --manifest "${MANIFEST}" \
  --region "${REGION}" \
  --center "${CENTER}" \
  --input-txt "${INPUT_TXT}" \
  --single-center-policy "${SINGLE_POLICY}" \
  --quiet-save-message \
  --output-json "${OUT_FP}" > /dev/null

echo "Prediction summary:"
"${PYTHON_BIN}" - "$OUT_FP" <<'PY'
import json
import sys

fp = sys.argv[1]
obj = json.loads(open(fp, "r", encoding="utf-8").read())

print(f"- mode: {obj.get('mode')}")
print(f"- policy: {obj.get('policy_used')} ({obj.get('strategy_name')})")
print(f"- predicted_class: {obj.get('predicted_class')}")
print("- probabilities:")
for k, v in (obj.get("probabilities") or {}).items():
    print(f"  - {k}: {v:.6f}")
PY

if [[ -n "${OUTPUT_JSON}" ]]; then
  echo "Saved full JSON: ${OUT_FP}"
else
  echo "Saved full JSON (default path): ${OUT_FP}"
fi
