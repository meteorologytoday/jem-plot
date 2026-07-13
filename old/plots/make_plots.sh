#!/bin/bash
# Generate all diagnostic plots for a simulation run.
#
# Usage:
#   bash plots/run_all_plots.sh <folder> [output_dir]
#       [--mean-batch-range START END]
#       [--ts-batch-range   START END]
#
# --mean-batch-range : batch range for mean plots (map, zonal mean, northward transport)
# --ts-batch-range   : batch range for timeseries plots
# Both default to all batches if not specified.
#
# Example:
#   bash plots/run_all_plots.sh output_T31/long_run output/plots \
#       --mean-batch-range 200 301 --ts-batch-range 0 301

set -euo pipefail

FOLDER="${1:?Usage: $0 <folder> [output_dir] [--mean-batch-range S E] [--ts-batch-range S E]}"
OUTPUT_DIR="${2:-output/plots}"
shift 2

MEAN_BATCH_ARGS=""
TS_BATCH_ARGS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mean-batch-range) MEAN_BATCH_ARGS="--batch-range $2 $3"; shift 3 ;;
        --ts-batch-range)   TS_BATCH_ARGS="--batch-range $2 $3";   shift 3 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

mkdir -p "$OUTPUT_DIR"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

run() {
    echo "Running: $*"
    python3 "$SCRIPT_DIR/$@"
}

# Mean plots
run plot_mean_map.py            --folder "$FOLDER" $MEAN_BATCH_ARGS --output "$OUTPUT_DIR/mean_map.png"
run plot_zonal_mean.py          --folder "$FOLDER" $MEAN_BATCH_ARGS --output "$OUTPUT_DIR/zonal_mean.png"
run plot_northward_transport.py --folder "$FOLDER" $MEAN_BATCH_ARGS --output "$OUTPUT_DIR/northward_transport_GLB.png"
run plot_northward_transport.py --folder "$FOLDER" $MEAN_BATCH_ARGS --lon-range 0 120 --output "$OUTPUT_DIR/northward_transport_ATL.png"

# Timeseries plots
run plot_oht_timeseries.py  --folder "$FOLDER" $TS_BATCH_ARGS --lon-range 0 120  --latitude 50 --output "$OUTPUT_DIR/oht_ost_timeseries.png"
run plot_timeseries.py      --folder "$FOLDER" $TS_BATCH_ARGS   --output "$OUTPUT_DIR/timeseries.png"

echo ""
echo "All plots saved to: $OUTPUT_DIR"
