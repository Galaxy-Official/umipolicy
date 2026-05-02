#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FUSION_METHOD=film exec "${SCRIPT_DIR}/run_eval_flexiv_tactile_linear_fusion.sh" "$@"
