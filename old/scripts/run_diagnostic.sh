#!/usr/bin/env bash
#
# Generic wrapper around plot_jem.py.
#
# Usage:
#   scripts/run_diagnostic.sh <subcommand> [options]
#
# Subcommands: list-vars, map, timeseries, zonal, profile
# Run 'scripts/run_diagnostic.sh <subcommand> --help' for subcommand options.
#
# Examples:
#   scripts/run_diagnostic.sh list-vars --run-dir jem_data/test-highres --component atm
#   scripts/run_diagnostic.sh map --run-dir jem_data/test-highres --component atm \
#       --variable temperature --time mean --level mean
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

if [[ $# -eq 0 ]]; then
    echo "Usage: $0 <subcommand> [options]" >&2
    echo "Subcommands: list-vars, map, timeseries, zonal, profile" >&2
    echo "Run '$0 <subcommand> --help' for subcommand-specific options." >&2
    exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
    echo "Error: python3 not found on PATH" >&2
    exit 1
fi

mkdir -p "$REPO_ROOT/figures"

exec python3 "$REPO_ROOT/plot_jem.py" "$@"
