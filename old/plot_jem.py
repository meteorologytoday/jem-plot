#!/usr/bin/env python3
"""Command-line diagnostic plotting tool for jax-esm (jem) output.

Run ``plot_jem.py <subcommand> --help`` for subcommand-specific options, or
see the README for example invocations.
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from jemdiag import data, plotting, selection  # noqa: E402


def _add_common_args(parser):
    parser.add_argument(
        "--run-dir", required=True, help="Path to a jem run output directory"
    )
    parser.add_argument(
        "--component", required=True, choices=data.COMPONENTS, help="jem component"
    )


def _add_plot_args(parser):
    parser.add_argument("--variable", required=True, help="Name of the variable to plot")
    parser.add_argument(
        "--output",
        help="Path to save the figure (PNG). Defaults to "
        "figures/<component>_<variable>_<subcommand>.png",
    )
    parser.add_argument("--title", help="Custom plot title")
    parser.add_argument("--cmap", help="Matplotlib colormap name (map/zonal plots)")
    parser.add_argument("--vmin", type=float, help="Color scale minimum (map/zonal plots)")
    parser.add_argument("--vmax", type=float, help="Color scale maximum (map/zonal plots)")


def _add_time_arg(parser):
    parser.add_argument(
        "--time",
        default="mean",
        help="Time index, 'mean', or 'start:end' slice (default: mean)",
    )


def _add_region_arg(parser):
    parser.add_argument(
        "--region",
        nargs=4,
        type=float,
        metavar=("LON_MIN", "LON_MAX", "LAT_MIN", "LAT_MAX"),
        help="Bounding box for the spatial average (requires real lon/lat coordinates)",
    )


def _default_output(args, subcommand):
    safe_var = args.variable.replace(".", "_")
    return Path("figures") / f"{args.component}_{safe_var}_{subcommand}.png"


def _save(fig, args, subcommand):
    output = Path(args.output) if args.output else _default_output(args, subcommand)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved figure to {output}")


def cmd_list_vars(args):
    for name in data.list_variables(args.run_dir, args.component):
        print(name)
    return 0


def cmd_map(args):
    da = data.load_variable(args.run_dir, args.component, args.variable)
    da2d = selection.reduce_to_map(da, args.time, args.level)
    fig, _ = plotting.plot_map(da2d, cmap=args.cmap, vmin=args.vmin, vmax=args.vmax, title=args.title)
    _save(fig, args, "map")
    return 0


def cmd_timeseries(args):
    da = data.load_variable(args.run_dir, args.component, args.variable)
    da1d = selection.reduce_to_timeseries(da, args.level, args.region)
    fig, _ = plotting.plot_timeseries(da1d, title=args.title)
    _save(fig, args, "timeseries")
    return 0


def cmd_zonal(args):
    da = data.load_variable(args.run_dir, args.component, args.variable)
    da2d = selection.reduce_to_zonal(da, args.time)
    fig, _ = plotting.plot_zonal_mean(
        da2d, cmap=args.cmap, vmin=args.vmin, vmax=args.vmax,
        invert_yaxis=args.invert_yaxis, title=args.title,
    )
    _save(fig, args, "zonal")
    return 0


def cmd_profile(args):
    da = data.load_variable(args.run_dir, args.component, args.variable)
    da1d = selection.reduce_to_profile(da, args.time, args.region)
    fig, _ = plotting.plot_profile(da1d, title=args.title)
    _save(fig, args, "profile")
    return 0


def build_parser():
    parser = argparse.ArgumentParser(description="Plot diagnostics from jax-esm (jem) output.")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    p = subparsers.add_parser("list-vars", help="List available variables for a component")
    _add_common_args(p)
    p.set_defaults(func=cmd_list_vars)

    p = subparsers.add_parser("map", help="Plot a horizontal (lat-lon) map")
    _add_common_args(p)
    _add_plot_args(p)
    _add_time_arg(p)
    p.add_argument(
        "--level",
        default="mean",
        help="Vertical-level index, slice, or 'mean' (default: mean; "
        "ignored if the variable has no vertical dimension)",
    )
    p.set_defaults(func=cmd_map)

    p = subparsers.add_parser("timeseries", help="Plot a domain-mean time series")
    _add_common_args(p)
    _add_plot_args(p)
    p.add_argument(
        "--level",
        default="mean",
        help="Vertical-level index, slice, or 'mean' (default: mean; "
        "ignored if the variable has no vertical dimension)",
    )
    _add_region_arg(p)
    p.set_defaults(func=cmd_timeseries)

    p = subparsers.add_parser("zonal", help="Plot a zonal-mean cross section (latitude vs. level/depth)")
    _add_common_args(p)
    _add_plot_args(p)
    _add_time_arg(p)
    p.add_argument("--invert-yaxis", action="store_true", help="Invert the vertical axis")
    p.set_defaults(func=cmd_zonal)

    p = subparsers.add_parser("profile", help="Plot a domain-mean vertical profile")
    _add_common_args(p)
    _add_plot_args(p)
    _add_time_arg(p)
    _add_region_arg(p)
    p.set_defaults(func=cmd_profile)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
