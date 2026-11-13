#!/usr/bin/env bash
#
# Generic wrapper around plot_jem.py.
#
# Usage:
#   scripts/run_diagnostic.sh <subcommand> [options]
#
# Subcommands: run, list-vars
# Run 'scripts/run_diagnostic.sh <subcommand> --help' for subcommand options.
#
# Examples:
#   scripts/run_diagnostic.sh run --sim config/sim/long_run.yaml --plot config/plot/default.yaml
#   scripts/run_diagnostic.sh list-vars --run-dir data/long_run --component atm
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

if [[ $# -eq 0 ]]; then
    echo "Usage: $0 <subcommand> [options]" >&2
    echo "Subcommands: run, list-vars" >&2
    echo "Run '$0 <subcommand> --help' for subcommand-specific options." >&2
    exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
    echo "Error: python3 not found on PATH" >&2
    exit 1
fi

exec python3 "$REPO_ROOT/plot_jem.py" "$@"
