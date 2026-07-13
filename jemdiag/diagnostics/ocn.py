"""
Ocean diagnostic plots.

Supported diag_types:
  map_2d          — horizontal map per variable in
                    config[diagnostics][ocn][map_2d][vars]. Uses a cartopy
                    projection when the grid has real lon/lat coordinates,
                    and falls back to a plain index-based map otherwise (see
                    jemdiag.grid.has_real_horizontal_coords).
  zonal_section   — latitude vs. depth cross-section
  profile         — domain-mean vertical profile
  timeseries      — area-weighted domain-mean time series
"""

from pathlib import Path
from typing import Callable

import numpy as np

from jemdiag import grid, plot_utils, selection


def run(diag_type: str, ds, config: dict, output_dir: Path) -> None:
    handler = HANDLERS.get(diag_type)
    if handler is None:
        return
    handler(ds, config, output_dir)


def _var_label(da) -> tuple[str, str]:
    """Return ``(title_name, units)`` derived from the DataArray's own attrs.

    Some jem long_names already start with "ocean ..." (e.g. "ocean
    potential temperature"); strip that so titles don't read
    "Ocean ocean potential temperature".
    """
    name = da.attrs.get("description") or da.attrs.get("long_name") or da.name
    name = str(name)
    if name.lower().startswith("ocean "):
        name = name[len("ocean "):]
    units = da.attrs.get("units", "")
    return name, units


def _run_map_2d(ds, config: dict, output_dir: Path) -> None:
    diag_cfg = config["diagnostics"]["ocn"].get("map_2d", {})
    out_cfg = config["output"]
    time_arg = diag_cfg.get("time", "mean")
    level_arg = diag_cfg.get("level", "mean")
    var_opts: dict = diag_cfg.get("var_opts", {})

    for varname in diag_cfg.get("vars", []):
        if varname not in ds:
            print(f"  [ocn/map_2d] skipping '{varname}' — not in dataset")
            continue
        da = ds[varname]
        try:
            da2d = selection.reduce_to_map(da, time_arg, level_arg)
        except (ValueError, KeyError) as exc:
            print(f"  [ocn/map_2d] skipping '{varname}': {exc}")
            continue

        x_dim, y_dim = grid.horizontal_dims(da2d)
        x, y, _ = grid.horizontal_coords(da2d, x_dim, y_dim)
        da2d = da2d.transpose(y_dim, x_dim)

        opts = var_opts.get(varname, {})
        name, units = _var_label(da)
        out_path = _out_path(output_dir, "ocn", "map_2d", varname, out_cfg["format"])
        common_kwargs = dict(
            cmap=opts.get("cmap"),
            units=opts.get("units", units),
            vmin=opts.get("vmin"),
            vmax=opts.get("vmax"),
            symmetric=opts.get("symmetric", False),
            n_levels=opts.get("n_levels", out_cfg.get("n_levels", 20)),
            dpi=out_cfg["dpi"],
        )
        if grid.has_real_horizontal_coords(da2d, x_dim, y_dim):
            plot_utils.global_map(
                data=da2d, lon=x, lat=y,
                title=f"Ocean {name}", output_path=out_path, **common_kwargs,
            )
        else:
            plot_utils.index_map(
                data=da2d, x=x, y=y, x_label=x_dim, y_label=y_dim,
                title=f"Ocean {name}", output_path=out_path, **common_kwargs,
            )
        print(f"  [ocn/map_2d] saved {out_path}")


def _run_zonal_section(ds, config: dict, output_dir: Path) -> None:
    diag_cfg = config["diagnostics"]["ocn"].get("zonal_section", {})
    out_cfg = config["output"]
    time_arg = diag_cfg.get("time", "mean")
    invert_yaxis = diag_cfg.get("invert_yaxis", True)
    var_opts: dict = diag_cfg.get("var_opts", {})

    for varname in diag_cfg.get("vars", []):
        if varname not in ds:
            print(f"  [ocn/zonal_section] skipping '{varname}' — not in dataset")
            continue
        da = ds[varname]
        try:
            da2d = selection.reduce_to_zonal(da, time_arg)
        except ValueError as exc:
            print(f"  [ocn/zonal_section] skipping '{varname}': {exc}")
            continue

        z_dim = grid.vertical_dim(da2d)
        y_dim = [d for d in da2d.dims if d != z_dim][0]
        da2d = da2d.transpose(z_dim, y_dim)

        opts = var_opts.get(varname, {})
        name, units = _var_label(da)
        out_path = _out_path(output_dir, "ocn", "zonal_section", varname, out_cfg["format"])
        plot_utils.zonal_section(
            data=da2d,
            y=da2d[y_dim].values,
            z=da2d[z_dim].values,
            y_label=y_dim,
            z_label=z_dim,
            title=f"Ocean {name} zonal mean",
            output_path=out_path,
            cmap=opts.get("cmap"),
            units=opts.get("units", units),
            vmin=opts.get("vmin"),
            vmax=opts.get("vmax"),
            symmetric=opts.get("symmetric", False),
            n_levels=opts.get("n_levels", out_cfg.get("n_levels", 20)),
            invert_yaxis=opts.get("invert_yaxis", invert_yaxis),
            dpi=out_cfg["dpi"],
        )
        print(f"  [ocn/zonal_section] saved {out_path}")


def _run_profile(ds, config: dict, output_dir: Path) -> None:
    diag_cfg = config["diagnostics"]["ocn"].get("profile", {})
    out_cfg = config["output"]
    time_arg = diag_cfg.get("time", "mean")
    region = diag_cfg.get("region")

    for varname in diag_cfg.get("vars", []):
        if varname not in ds:
            print(f"  [ocn/profile] skipping '{varname}' — not in dataset")
            continue
        da = ds[varname]
        try:
            da1d = selection.reduce_to_profile(da, time_arg, region)
        except ValueError as exc:
            print(f"  [ocn/profile] skipping '{varname}': {exc}")
            continue

        z_dim = grid.vertical_dim(da1d)
        name, units = _var_label(da)
        out_path = _out_path(output_dir, "ocn", "profile", varname, out_cfg["format"])
        plot_utils.profile(
            data=da1d,
            z=da1d[z_dim].values,
            z_label=z_dim,
            title=f"Ocean {name} profile",
            output_path=out_path,
            units=units,
            dpi=out_cfg["dpi"],
        )
        print(f"  [ocn/profile] saved {out_path}")


def _run_timeseries(ds, config: dict, output_dir: Path) -> None:
    diag_cfg = config["diagnostics"]["ocn"].get("timeseries", {})
    out_cfg = config["output"]
    level_arg = diag_cfg.get("level", "mean")
    region = diag_cfg.get("region")
    running_mean_steps = diag_cfg.get("running_mean_steps")

    for varname in diag_cfg.get("vars", []):
        if varname not in ds:
            print(f"  [ocn/timeseries] skipping '{varname}' — not in dataset")
            continue
        da = ds[varname]
        try:
            da1d = selection.reduce_to_timeseries(da, level_arg, region)
        except ValueError as exc:
            print(f"  [ocn/timeseries] skipping '{varname}': {exc}")
            continue

        name, units = _var_label(da)
        out_path = _out_path(output_dir, "ocn", "timeseries", varname, out_cfg["format"])
        plot_utils.time_series(
            times=da1d["time"].values,
            values=np.asarray(da1d.values, dtype=float),
            title=f"Ocean domain-mean {name}",
            ylabel=units or str(varname),
            output_path=out_path,
            dpi=out_cfg["dpi"],
            running_mean_steps=running_mean_steps,
        )
        print(f"  [ocn/timeseries] saved {out_path}")


def _out_path(output_dir: Path, component: str, diag_type: str, varname: str, fmt: str) -> Path:
    safe_var = varname.replace(".", "_")
    return output_dir / component / diag_type / f"{safe_var}.{fmt}"


HANDLERS: dict[str, Callable] = {
    "map_2d": _run_map_2d,
    "zonal_section": _run_zonal_section,
    "profile": _run_profile,
    "timeseries": _run_timeseries,
}
