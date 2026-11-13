"""
Atmosphere diagnostic plots.

Supported diag_types:
  map_2d          — global (cartopy) map per variable in
                    config[diagnostics][atm][map_2d][vars]
  zonal_section   — latitude vs. level cross-section
  profile         — domain-mean vertical profile
  timeseries      — area-weighted domain-mean time series
  zonal_mean      — zonally-averaged 1D profile vs. latitude, for a surface
                    field with no vertical dimension. Supports the derived
                    pseudo-variable `precipitation` (`condensation.precls`
                    + `convection.precnv` — jem reports large-scale and
                    convective precipitation separately, with no combined
                    total variable)
  heat_transport  — meridional atmospheric heat transport (PW). Two
                    methods, picked via config `method` (default
                    `dynamical`):
                      dynamical — from time-averaged `v_wind`/`temperature`/
                        `normalized_surface_pressure`; mean-circulation
                        transport only, no transient-eddy term.
                      radiative — top-of-atmosphere/surface residual
                        ("indirect") method from `shortwave_rad.ftop`/
                        `longwave_rad.ftop`/`-surface_flux.hfluxn.1`;
                        captures the total (mean + eddy) transport. Uses
                        `hfluxn.1` alone (not `hfluxn.0`, and not a sum of
                        the two): this run's surface exchange is only
                        actually driven through the `.1` tile (confirmed
                        against `data/long_run` — `fakelnd`'s
                        `total_heat_flux` is the exact sign-flipped mirror
                        of `hfluxn.1`, `.0` isn't part of this run's real
                        surface energy budget). `hfluxn.1` itself is
                        negated before use: it's stored positive-upward
                        (atm's own convention), but
                        :func:`compute.atmosphere_heat_transport_radiative`
                        needs a positive-*downward* surface flux (verified
                        against `data/long_run` — without the negation the
                        implied transport is ~27 PW at the pole where it
                        should be ~0; negated, it's ~1 PW).
"""

from pathlib import Path
from typing import Callable

import numpy as np

from jemdiag import compute, grid, plot_utils, selection


def run(diag_type: str, ds, config: dict, output_dir: Path) -> None:
    handler = HANDLERS.get(diag_type)
    if handler is None:
        return
    handler(ds, config, output_dir)


def _var_label(da) -> tuple[str, str]:
    """Return ``(title_name, units)`` derived from the DataArray's own attrs."""
    name = da.attrs.get("description") or da.attrs.get("long_name") or da.name
    units = da.attrs.get("units", "")
    return str(name), units


def _run_map_2d(ds, config: dict, output_dir: Path) -> None:
    diag_cfg = config["diagnostics"]["atm"].get("map_2d", {})
    out_cfg = config["output"]
    grid_cfg = config.get("grids", {}).get("atm", {})
    projection = grid_cfg.get("projection", "PlateCarree")
    central_longitude = plot_utils.resolve_central_longitude(
        diag_cfg, config["diagnostics"]
    )
    coastlines = plot_utils.resolve_coastlines(diag_cfg, config["diagnostics"])
    time_args = selection.resolve_time_spec(diag_cfg, config["diagnostics"])
    level_arg = diag_cfg.get("level", "mean")
    var_opts: dict = diag_cfg.get("var_opts", {})

    for varname in diag_cfg.get("vars", []):
        da = _resolve_var(ds, varname)
        if da is None:
            print(f"  [atm/map_2d] skipping '{varname}' — not in dataset")
            continue
        opts = var_opts.get(varname, {})
        name, units = _var_label(da)

        for time_arg in time_args:
            try:
                da2d = selection.reduce_to_map(da, time_arg, level_arg)
            except (ValueError, KeyError) as exc:
                print(f"  [atm/map_2d] skipping '{varname}': {exc}")
                continue

            x_dim, y_dim = grid.horizontal_dims(da2d)
            lon, lat, _ = grid.horizontal_coords(da2d, x_dim, y_dim)
            da2d = da2d.transpose(y_dim, x_dim)

            time_label = selection.time_label(time_arg)
            suffix = time_label if len(time_args) > 1 else None
            title = f"Atmos {name} ({time_label})"
            out_path = _out_path(
                output_dir, "atm", "map_2d", varname, out_cfg["format"], time_suffix=suffix
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
            print(f"  [atm/map_2d] saved {out_path}")


def _run_zonal_section(ds, config: dict, output_dir: Path) -> None:
    diag_cfg = config["diagnostics"]["atm"].get("zonal_section", {})
    out_cfg = config["output"]
    time_args = selection.resolve_time_spec(diag_cfg, config["diagnostics"])
    invert_yaxis = diag_cfg.get("invert_yaxis", False)
    var_opts: dict = diag_cfg.get("var_opts", {})

    for varname in diag_cfg.get("vars", []):
        if varname not in ds:
            print(f"  [atm/zonal_section] skipping '{varname}' — not in dataset")
            continue
        da = ds[varname]
        opts = var_opts.get(varname, {})
        name, units = _var_label(da)

        for time_arg in time_args:
            try:
                da2d = selection.reduce_to_zonal(da, time_arg)
            except ValueError as exc:
                print(f"  [atm/zonal_section] skipping '{varname}': {exc}")
                continue

            z_dim = grid.vertical_dim(da2d)
            y_dim = [d for d in da2d.dims if d != z_dim][0]
            da2d = da2d.transpose(z_dim, y_dim)

            time_label = selection.time_label(time_arg)
            suffix = time_label if len(time_args) > 1 else None
            title = f"Atmos {name} zonal mean ({time_label})"
            out_path = _out_path(
                output_dir, "atm", "zonal_section", varname, out_cfg["format"], time_suffix=suffix
            )
            plot_utils.zonal_section(
                data=da2d,
                y=da2d[y_dim].values,
                z=da2d[z_dim].values,
                y_label=y_dim,
                z_label=z_dim,
                title=title,
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
            print(f"  [atm/zonal_section] saved {out_path}")


def _run_profile(ds, config: dict, output_dir: Path) -> None:
    diag_cfg = config["diagnostics"]["atm"].get("profile", {})
    out_cfg = config["output"]
    time_args = selection.resolve_time_spec(diag_cfg, config["diagnostics"])
    region = diag_cfg.get("region")

    for varname in diag_cfg.get("vars", []):
        if varname not in ds:
            print(f"  [atm/profile] skipping '{varname}' — not in dataset")
            continue
        da = ds[varname]
        name, units = _var_label(da)

        for time_arg in time_args:
            try:
                da1d = selection.reduce_to_profile(da, time_arg, region)
            except ValueError as exc:
                print(f"  [atm/profile] skipping '{varname}': {exc}")
                continue

            z_dim = grid.vertical_dim(da1d)
            time_label = selection.time_label(time_arg)
            suffix = time_label if len(time_args) > 1 else None
            title = f"Atmos {name} profile ({time_label})"
            out_path = _out_path(
                output_dir, "atm", "profile", varname, out_cfg["format"], time_suffix=suffix
            )
            plot_utils.profile(
                data=da1d,
                z=da1d[z_dim].values,
                z_label=z_dim,
                title=title,
                output_path=out_path,
                units=units,
                dpi=out_cfg["dpi"],
            )
            print(f"  [atm/profile] saved {out_path}")


def _run_timeseries(ds, config: dict, output_dir: Path) -> None:
    diag_cfg = config["diagnostics"]["atm"].get("timeseries", {})
    out_cfg = config["output"]
    level_arg = diag_cfg.get("level", "mean")
    region = diag_cfg.get("region")
    running_mean_steps = diag_cfg.get("running_mean_steps")

    for varname in diag_cfg.get("vars", []):
        if varname not in ds:
            print(f"  [atm/timeseries] skipping '{varname}' — not in dataset")
            continue
        da = ds[varname]
        try:
            da1d = selection.reduce_to_timeseries(da, level_arg, region)
        except ValueError as exc:
            print(f"  [atm/timeseries] skipping '{varname}': {exc}")
            continue

        name, units = _var_label(da)
        out_path = _out_path(output_dir, "atm", "timeseries", varname, out_cfg["format"])
        times, time_label = selection.time_axis(da1d)
        plot_utils.time_series(
            times=times,
            values=np.asarray(da1d.values, dtype=float),
            title=f"Atmos domain-mean {name}",
            ylabel=units or str(varname),
            output_path=out_path,
            dpi=out_cfg["dpi"],
            running_mean_steps=running_mean_steps,
            xlabel=time_label,
        )
        print(f"  [atm/timeseries] saved {out_path}")


def _resolve_var(ds, varname):
    """Return the DataArray for ``varname``, or ``None`` if unavailable.

    Handles ``precipitation`` as the derived sum of large-scale and
    convective precip (``condensation.precls`` + ``convection.precnv``),
    since jem reports those separately with no combined total variable.
    """
    if varname == "precipitation":
        required = ("condensation.precls", "convection.precnv")
        if not all(v in ds for v in required):
            return None
        total = ds["condensation.precls"] + ds["convection.precnv"]
        total.name = "precipitation"
        total.attrs = {"long_name": "precipitation"}
        return total
    return ds.get(varname)


def _run_zonal_mean(ds, config: dict, output_dir: Path) -> None:
    diag_cfg = config["diagnostics"]["atm"].get("zonal_mean", {})
    out_cfg = config["output"]
    time_args = selection.resolve_time_spec(diag_cfg, config["diagnostics"])

    for varname in diag_cfg.get("vars", []):
        da = _resolve_var(ds, varname)
        if da is None:
            print(f"  [atm/zonal_mean] skipping '{varname}' — not in dataset")
            continue
        name, units = _var_label(da)
        _, y_dim = grid.horizontal_dims(da)

        for time_arg in time_args:
            da1d = selection.reduce_to_meridional_profile(da, time_arg)

            time_label = selection.time_label(time_arg)
            suffix = time_label if len(time_args) > 1 else None
            title = f"Atmos {name} zonal mean ({time_label})"
            out_path = _out_path(
                output_dir, "atm", "zonal_mean", varname, out_cfg["format"], time_suffix=suffix
            )
            plot_utils.meridional_profile(
                y=da1d[y_dim].values,
                values=np.asarray(da1d.values, dtype=float),
                y_label=y_dim,
                title=title,
                output_path=out_path,
                units=units,
                dpi=out_cfg["dpi"],
            )
            print(f"  [atm/zonal_mean] saved {out_path}")


def _run_heat_transport(ds, config: dict, output_dir: Path) -> None:
    diag_cfg = config["diagnostics"]["atm"].get("heat_transport")
    if diag_cfg is None:
        return
    out_cfg = config["output"]
    time_args = selection.resolve_time_spec(diag_cfg, config["diagnostics"])
    method = diag_cfg.get("method", "dynamical")

    if method == "dynamical":
        required = ("v_wind", "temperature", "normalized_surface_pressure")
    elif method == "radiative":
        required = ("shortwave_rad.ftop", "longwave_rad.ftop", "surface_flux.hfluxn.1")
    else:
        print(
            f"  [atm/heat_transport] skipping — unknown method '{method}' "
            "(expected 'dynamical' or 'radiative')"
        )
        return

    missing = [name for name in required if name not in ds]
    if missing:
        print(f"  [atm/heat_transport] skipping — missing {missing} in dataset")
        return

    if method == "dynamical":
        x_dim, y_dim = grid.horizontal_dims(ds["v_wind"])
        z_dim = grid.vertical_dim(ds["v_wind"])
        if z_dim is None:
            print("  [atm/heat_transport] skipping — 'v_wind' has no vertical dimension")
            return
    else:
        x_dim, y_dim = grid.horizontal_dims(ds["shortwave_rad.ftop"])

    for time_arg in time_args:
        try:
            if method == "dynamical":
                v = selection.select_along(ds["v_wind"], "time", time_arg, by_label=True)
                temp = selection.select_along(ds["temperature"], "time", time_arg, by_label=True)
                ps = selection.select_along(ds["normalized_surface_pressure"], "time", time_arg, by_label=True)
                aht = compute.atmosphere_heat_transport(v, temp, ps, x_dim, y_dim, z_dim)
            else:
                toa_net = selection.select_along(
                    ds["shortwave_rad.ftop"], "time", time_arg, by_label=True
                ) - selection.select_along(ds["longwave_rad.ftop"], "time", time_arg, by_label=True)
                surface_net = -selection.select_along(ds["surface_flux.hfluxn.1"], "time", time_arg, by_label=True)
                aht = compute.atmosphere_heat_transport_radiative(toa_net, surface_net, x_dim, y_dim)
        except ValueError as exc:
            print(f"  [atm/heat_transport] skipping: {exc}")
            continue

        time_label = selection.time_label(time_arg)
        suffix = time_label if len(time_args) > 1 else None
        title = f"Meridional atmospheric heat transport ({method}, {time_label})"
        out_path = _out_path(
            output_dir, "atm", "heat_transport", "aht", out_cfg["format"], time_suffix=suffix
        )
        plot_utils.meridional_transport(
            y=aht[y_dim].values,
            values=np.asarray(aht.values, dtype=float),
            y_label=y_dim,
            title=title,
            output_path=out_path,
            units="PW",
            dpi=out_cfg["dpi"],
        )
        print(f"  [atm/heat_transport] saved {out_path}")


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
    "zonal_section": _run_zonal_section,
    "profile": _run_profile,
    "timeseries": _run_timeseries,
    "zonal_mean": _run_zonal_mean,
    "heat_transport": _run_heat_transport,
}
