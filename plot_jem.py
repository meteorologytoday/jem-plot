#!/usr/bin/env python3
"""
plot_jem.py — config-driven diagnostic plotting for jax-esm (jem) output.

Usage
-----
    python plot_jem.py run --sim config/sim/long_run.yaml \\
        --plot config/plot/default.yaml --components atm ocn --diag-types map_2d

    python plot_jem.py list-vars --run-dir data/long_run --component atm

See the README for the config file schemas and more examples.
"""

import argparse
import copy
import sys
from pathlib import Path

from jemdiag import data_loader
from jemdiag.config import load_config
from jemdiag.diagnostics import atm, lnd, ocn, seaice

# Registry: component name -> module with a HANDLERS dict and run() function.
COMPONENT_MODULES = {
    "atm": atm,
    "ocn": ocn,
    "lnd": lnd,
    "seaice": seaice,
}

ALL_DIAG_TYPES = [
    "map_2d", "zonal_section", "profile", "timeseries", "zonal_mean",
    "overturning_streamfunction", "heat_transport",
]


def _add_run_subparser(subparsers):
    p = subparsers.add_parser(
        "run", help="Generate figures from a simulation config and a plot config"
    )
    p.add_argument(
        "--sim", required=True, type=Path, metavar="PATH",
        help=(
            "Simulation YAML config: `experiment`/`data`, plus optional "
            "run-specific `diagnostics` overrides (e.g. config/sim/long_run.yaml)."
        ),
    )
    p.add_argument(
        "--plot", required=True, type=Path, metavar="PATH",
        help=(
            "Plot YAML config: `grids`/`diagnostics`/`output` "
            "(e.g. config/plot/default.yaml)."
        ),
    )
    p.add_argument(
        "--components", nargs="+", default=list(COMPONENT_MODULES),
        choices=list(COMPONENT_MODULES), metavar="COMPONENT",
        help=f"jem components to process: {list(COMPONENT_MODULES)}. Default: all.",
    )
    p.add_argument(
        "--diag-types", nargs="+", default=ALL_DIAG_TYPES, choices=ALL_DIAG_TYPES,
        metavar="TYPE",
        help=(
            f"Diagnostic plot types: {ALL_DIAG_TYPES}. Default: all. "
            "Each component only produces types it supports."
        ),
    )
    p.add_argument(
        "--run-dir", type=Path, default=None, metavar="DIR",
        help="Override data.run_dir from the simulation config.",
    )
    p.add_argument(
        "--exp-name", default=None, metavar="NAME",
        help="Override experiment.name from the simulation config (used in output paths).",
    )
    p.add_argument(
        "--output-dir", type=Path, default=None, metavar="DIR",
        help="Override output.base_dir from the plot config.",
    )
    p.add_argument(
        "--vars", nargs="+", default=None, metavar="VAR",
        help="Restrict to these variable names across all selected components/types.",
    )
    p.set_defaults(func=cmd_run)


def _add_list_vars_subparser(subparsers):
    p = subparsers.add_parser("list-vars", help="List available variables for a component")
    p.add_argument("--run-dir", required=True, help="Path to a jem run output directory")
    p.add_argument(
        "--component", required=True, choices=data_loader.COMPONENTS, help="jem component"
    )
    p.set_defaults(func=cmd_list_vars)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Plot diagnostics from jax-esm (jem) output.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Everything the plot config describes, for one simulation
  python plot_jem.py run --sim config/sim/long_run.yaml \\
      --plot config/plot/default.yaml

  # Only atm maps, for fast iteration
  python plot_jem.py run --sim config/sim/long_run.yaml \\
      --plot config/plot/default.yaml --components atm --diag-types map_2d

  # Single variable during development
  python plot_jem.py run --sim config/sim/long_run.yaml \\
      --plot config/plot/default.yaml --components atm --diag-types map_2d \\
      --vars temperature

  # Point at a different run directory without a separate simulation config
  python plot_jem.py run --sim config/sim/long_run.yaml \\
      --plot config/plot/default.yaml --run-dir /path/to/other/run \\
      --exp-name my_run

  python plot_jem.py list-vars --run-dir data/long_run --component atm
""",
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)
    _add_run_subparser(subparsers)
    _add_list_vars_subparser(subparsers)
    return parser


def apply_var_filter(config: dict, vars_filter: list[str] | None) -> dict:
    """Restrict variable lists in config to those in vars_filter."""
    if not vars_filter:
        return config
    cfg = copy.deepcopy(config)
    diag = cfg.get("diagnostics", {})
    vset = set(vars_filter)
    for comp_cfg in diag.values():
        if not isinstance(comp_cfg, dict):
            continue  # e.g. the shared top-level `diagnostics.time` default
        for dtype_cfg in comp_cfg.values():
            if isinstance(dtype_cfg, dict) and "vars" in dtype_cfg:
                dtype_cfg["vars"] = [v for v in dtype_cfg["vars"] if v in vset]
    return cfg


def cmd_list_vars(args) -> int:
    for name in data_loader.list_variables(args.run_dir, args.component):
        print(name)
    return 0


def cmd_run(args) -> int:
    config = load_config(args.sim, args.plot)

    if args.run_dir is not None:
        config["data"]["run_dir"] = str(args.run_dir)
    if args.exp_name is not None:
        config["experiment"]["name"] = args.exp_name

    output_root = args.output_dir or Path(config["output"]["base_dir"])
    exp_name = config["experiment"]["name"]
    config = apply_var_filter(config, args.vars)

    print(f"Loading data for experiment '{exp_name}' from {config['data']['run_dir']} ...")
    try:
        data = data_loader.load_all(
            config, diag_types=args.diag_types, components=args.components
        )
    except FileNotFoundError as exc:
        print(f"Error loading data: {exc}", file=sys.stderr)
        return 1

    output_dir = output_root / exp_name

    for comp_name in args.components:
        if comp_name not in data:
            print(f"[{comp_name}] skipping — no data files found")
            continue
        module = COMPONENT_MODULES[comp_name]
        for diag_type in args.diag_types:
            if diag_type not in module.HANDLERS:
                continue
            print(f"Running {comp_name} / {diag_type} ...")
            module.run(diag_type, data[comp_name], config, output_dir)

    print(f"\nDone. Output written to: {output_dir}")
    return 0


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
