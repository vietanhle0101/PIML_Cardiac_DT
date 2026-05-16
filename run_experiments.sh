#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-config_ms.yaml}"

case "${1:-help}" in
  train-pinn)
    python main_training.py \
      --config "$CONFIG" \
      --checkpoint outputs/mlp_pinn.pt
    ;;

  train-data-only)
    python main_training.py \
      --config "$CONFIG" \
      --checkpoint outputs/mlp_data_only.pt \
      --data-weight 1.0 \
      --physics-weight 0.0 \
      --boundary-weight 0.0 \
      --initial-weight 0.0
    ;;

  evaluate-pinn)
    python evaluate_pinn.py \
      --config "$CONFIG" \
      --checkpoint outputs/mlp_pinn.pt \
      --output outputs/pinn_time_slices.png
    ;;

  evaluate-data-only)
    python evaluate_pinn.py \
      --config "$CONFIG" \
      --checkpoint outputs/mlp_data_only.pt \
      --output outputs/data_only_time_slices.png
    ;;

  collocation)
    python evaluate_pinn.py \
      --config "$CONFIG" \
      --checkpoint outputs/mlp_pinn.pt \
      --output outputs/pinn_time_slices.png \
      --show-collocation \
      --collocation-output outputs/collocation_points.png
    ;;

  help|*)
    echo "Usage: $0 {train-pinn|train-data-only|evaluate-pinn|evaluate-data-only|collocation}"
    echo
    echo "Optional: CONFIG=path/to/config.yaml $0 train-pinn"
    ;;
esac
