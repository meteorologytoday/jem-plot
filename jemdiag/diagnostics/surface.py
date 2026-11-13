"""
Diagnostic plots shared by the 2D surface components, ``lnd`` and ``seaice``.

Both live on the atm horizontal grid with real ``lon``/``lat`` coordinates.
``seaice`` has no vertical dimension; ``lnd``'s soil-water variables carry
a soil ``layer`` dim, reduced by the diag_type's ``level`` setting
(``"mean"`` if omitted) like any other vertical dim. Supported diag_types:

  map_2d      — global (cartopy) map per variable in
                config[diagnostics][<component>][map_2d][vars]
  timeseries  — area-weighted domain-mean time series

Each component module (:mod:`.lnd`, :mod:`.seaice`) is a thin
:func:`make_component` call, so the two can't drift apart.
"""

from functools import partial
from pathlib import Path
from typing import Callable

import numpy as np

from jemdiag import grid, plot_utils, selection


def _var_label(da) -> tuple[str, str]:
    name = da.attrs.get("description") or da.attrs.get("long_name") or da.name
    units = da.attrs.get("units", "")
    return str(name), units


def _run_map_2d(component: str, title_prefix: str, ds, config: dict, output_dir: Path) -> None:
    diag_cfg = config["diagnostics"][component].get("map_2d", {})
    out_cfg = config["output"]
    grid_cfg = config.get("grids", {}).get(component, {})
    projection = grid_cfg.get("projection", "PlateCarree")
    central_longitude = plot_utils.resolve_central_longitude(
        diag_cfg, config["diagnostics"]
    )
    coastlines = plot_utils.resolve_coastlines(diag_cfg, config["diagnostics"])
    time_args = selection.resolve_time_spec(diag_cfg, config["diagnostics"])
    level_arg = diag_cfg.get("level", "mean")
    var_opts: dict = diag_cfg.get("var_opts", {})
    tag = f"[{component}/map_2d]"

    for varname in diag_cfg.get("vars", []):
        if varname not in ds:
            print(f"  {tag} skipping '{varname}' — not in dataset")
            continue
        da = ds[varname]
        opts = var_opts.get(varname, {})
        name, units = _var_label(da)

        for time_arg in time_args:
            try:
                da2d = selection.reduce_to_map(da, time_arg, level_arg)
            except (ValueError, KeyError) as exc:
                print(f"  {tag} skipping '{varname}': {exc}")
                continue

            x_dim, y_dim = grid.horizontal_dims(da2d)
            lon, lat, _ = grid.horizontal_coords(da2d, x_dim, y_dim)
            da2d = da2d.transpose(y_dim, x_dim)

            time_label = selection.time_label(time_arg)
            suffix = time_label if len(time_args) > 1 else None
            title = f"{title_prefix} {name} ({time_label})"
            out_path = _out_path(
                output_dir, component, "map_2d", varname, out_cfg["format"], time_suffix=suffix
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
            print(f"  {tag} saved {out_path}")


def _run_timeseries(component: str, title_prefix: str, ds, config: dict, output_dir: Path) -> None:
    diag_cfg = config["diagnostics"][component].get("timeseries", {})
    out_cfg = config["output"]
    level_arg = diag_cfg.get("level", "mean")
    region = diag_cfg.get("region")
    running_mean_steps = diag_cfg.get("running_mean_steps")
    tag = f"[{component}/timeseries]"

    for varname in diag_cfg.get("vars", []):
        if varname not in ds:
            print(f"  {tag} skipping '{varname}' — not in dataset")
            continue
        da = ds[varname]
        try:
            da1d = selection.reduce_to_timeseries(da, level_arg, region)
        except ValueError as exc:
            print(f"  {tag} skipping '{varname}': {exc}")
            continue

        name, units = _var_label(da)
        out_path = _out_path(output_dir, component, "timeseries", varname, out_cfg["format"])
        times, time_label = selection.time_axis(da1d)
        plot_utils.time_series(
            times=times,
            values=np.asarray(da1d.values, dtype=float),
            title=f"{title_prefix} domain-mean {name}",
            ylabel=units or str(varname),
            output_path=out_path,
            dpi=out_cfg["dpi"],
            running_mean_steps=running_mean_steps,
            xlabel=time_label,
        )
        print(f"  {tag} saved {out_path}")


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


def make_component(component: str, title_prefix: str) -> tuple[Callable, dict[str, Callable]]:
    """Return ``(run, HANDLERS)`` for a surface component.

    *component* is the config/output key (``"lnd"``); *title_prefix* starts
    every figure title (``"Land"``).
    """
    handlers: dict[str, Callable] = {
        "map_2d": partial(_run_map_2d, component, title_prefix),
        "timeseries": partial(_run_timeseries, component, title_prefix),
    }

    def run(diag_type: str, ds, config: dict, output_dir: Path) -> None:
        handler = handlers.get(diag_type)
        if handler is None:
            return
        handler(ds, config, output_dir)

    return run, handlers
