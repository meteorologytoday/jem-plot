"""
fakelnd diagnostic plots.

fakelnd has real 2D curvilinear coordinates (``longitude2D``/``latitude2D``,
matching the atm lon/lat grid) but no vertical dimension, so only these
diag_types apply:

  map_2d      — global (cartopy) map per variable in
                config[diagnostics][fakelnd][map_2d][vars]
  timeseries  — area-weighted domain-mean time series
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
    name = da.attrs.get("description") or da.attrs.get("long_name") or da.name
    units = da.attrs.get("units", "")
    return str(name), units


def _run_map_2d(ds, config: dict, output_dir: Path) -> None:
    diag_cfg = config["diagnostics"]["fakelnd"].get("map_2d", {})
    out_cfg = config["output"]
    grid_cfg = config.get("grids", {}).get("fakelnd", {})
    projection = grid_cfg.get("projection", "PlateCarree")
    central_longitude = plot_utils.resolve_central_longitude(
        diag_cfg, config["diagnostics"]
    )
    coastlines = plot_utils.resolve_coastlines(diag_cfg, config["diagnostics"])
    time_args = selection.resolve_time_spec(diag_cfg, config["diagnostics"])
    var_opts: dict = diag_cfg.get("var_opts", {})

    for varname in diag_cfg.get("vars", []):
        if varname not in ds:
            print(f"  [fakelnd/map_2d] skipping '{varname}' — not in dataset")
            continue
        da = ds[varname]
        opts = var_opts.get(varname, {})
        name, units = _var_label(da)

        for time_arg in time_args:
            try:
                da2d = selection.reduce_to_map(da, time_arg, "mean")
            except (ValueError, KeyError) as exc:
                print(f"  [fakelnd/map_2d] skipping '{varname}': {exc}")
                continue

            x_dim, y_dim = grid.horizontal_dims(da2d)
            lon, lat, _ = grid.horizontal_coords(da2d, x_dim, y_dim)
            da2d = da2d.transpose(y_dim, x_dim)

            time_label = selection.time_label(time_arg)
            suffix = time_label if len(time_args) > 1 else None
            title = f"fakelnd {name} ({time_label})"
            out_path = _out_path(
                output_dir, "fakelnd", "map_2d", varname, out_cfg["format"], time_suffix=suffix
            )
            plot_utils.global_map(
                data=da2d,
                lon=lon,
                lat=lat,
                title=title,
                output_path=out_path,
                cmap=opts.get("cmap"),
                units=opts.get("units", units),
                vmin=opts.get("vmin"),
                vmax=opts.get("vmax"),
                symmetric=opts.get("symmetric", False),
                n_levels=opts.get("n_levels", out_cfg.get("n_levels", 20)),
                dpi=out_cfg["dpi"],
                projection=projection,
                central_longitude=central_longitude,
                coastlines=coastlines,
            )
            print(f"  [fakelnd/map_2d] saved {out_path}")


def _run_timeseries(ds, config: dict, output_dir: Path) -> None:
    diag_cfg = config["diagnostics"]["fakelnd"].get("timeseries", {})
    out_cfg = config["output"]
    region = diag_cfg.get("region")
    running_mean_steps = diag_cfg.get("running_mean_steps")

    for varname in diag_cfg.get("vars", []):
        if varname not in ds:
            print(f"  [fakelnd/timeseries] skipping '{varname}' — not in dataset")
            continue
        da = ds[varname]
        try:
            da1d = selection.reduce_to_timeseries(da, "mean", region)
        except ValueError as exc:
            print(f"  [fakelnd/timeseries] skipping '{varname}': {exc}")
            continue

        name, units = _var_label(da)
        out_path = _out_path(output_dir, "fakelnd", "timeseries", varname, out_cfg["format"])
        plot_utils.time_series(
            times=da1d["time"].values,
            values=np.asarray(da1d.values, dtype=float),
            title=f"fakelnd domain-mean {name}",
            ylabel=units or str(varname),
            output_path=out_path,
            dpi=out_cfg["dpi"],
            running_mean_steps=running_mean_steps,
        )
        print(f"  [fakelnd/timeseries] saved {out_path}")


def _out_path(
    output_dir: Path,
    component: str,
    diag_type: str,
    varname: str,
    fmt: str,
    time_suffix: str | None = None,
) -> Path:
    safe_var = varname.replace(".", "_")
    if time_suffix:
        safe_var = f"{safe_var}_{time_suffix}"
    return output_dir / component / diag_type / f"{safe_var}.{fmt}"


HANDLERS: dict[str, Callable] = {
    "map_2d": _run_map_2d,
    "timeseries": _run_timeseries,
}
