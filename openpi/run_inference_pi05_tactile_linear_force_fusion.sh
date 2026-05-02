#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_NAME="${CONFIG_NAME:-pi05_simple_sorting_tactile_linear_force_fusion}" \
POLICY_CONFIG="${POLICY_CONFIG:-pi05_simple_sorting_tactile_linear_force_fusion}" \
FORCE_PREDICT=true \
FORCE_GUIDE=true \
exec "${SCRIPT_DIR}/run_inference_pi05_tactile_linear_fusion.sh" "$@"
