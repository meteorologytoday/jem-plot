"""
Shared plotting utilities for jem diagnostic figures.

Horizontal maps are produced with cartopy (for a correct geographic
projection) whenever the underlying grid has real coordinates (``atm``,
``fakelnd``); no coastlines or other map features are drawn. ``ocn``
currently has no coordinate values at all (see :mod:`jemdiag.data_loader`),
so its maps fall back to :func:`index_map`, a plain (non-georeferenced)
``pcolormesh`` — once a future jem version adds real ocean coordinates,
:mod:`jemdiag.diagnostics.ocn` will pick :func:`global_map` automatically
(see :func:`jemdiag.grid.has_real_horizontal_coords`), with no config change
required.

Diagnostic modules should call these helpers rather than constructing
figures directly so that style changes propagate everywhere at once.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend for headless / SSH sessions
import cartopy.crs as ccrs
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

# Default colormaps keyed by variable name. Diagnostic modules fall back to
# "viridis" for variables not listed here. Overridable per-variable via
# config `var_opts`.
DEFAULT_CMAPS: dict[str, str] = {
    # atmosphere
    "temperature": "RdBu_r",
    "specific_humidity": "Blues",
    "humidity.rh": "Blues",
    "humidity.qsat": "Blues",
    "u_wind": "RdBu_r",
    "v_wind": "RdBu_r",
    "geopotential": "viridis",
    "shortwave_rad.rsds": "YlOrRd",
    "shortwave_rad.ftop": "YlOrRd",
    "longwave_rad.ftop": "YlOrRd_r",
    "surface_flux.tsfc": "RdBu_r",
    "surface_flux.tskin": "RdBu_r",
    # ocean
    "temp": "RdBu_r",
    "salt": "PuBu",
    "sea_surface_temperature": "RdBu_r",
    "sea_surface_salinity": "PuBu",
    "heat_flux": "RdBu_r",
    "freshwater_flux": "BrBG",
    # fakelnd
    "mixed_layer_depth": "viridis",
    "total_heat_flux": "RdBu_r",
}


# ── Projection factory ────────────────────────────────────────────────────────

def get_projection(name: str, **kwargs) -> ccrs.CRS:
    """Return a cartopy CRS by name (from config `grids.<component>.projection`)."""
    projections: dict[str, type] = {
        "PlateCarree": ccrs.PlateCarree,
    }
    if name not in projections:
        raise ValueError(
            f"Unknown projection {name!r}. Choose from: {sorted(projections)}"
        )
    return projections[name](**kwargs)


# ── Map figure builders ───────────────────────────────────────────────────────

def global_map(
    data: xr.DataArray,
    lon: np.ndarray,
    lat: np.ndarray,
    title: str,
    output_path: str | Path,
    cmap: str | None = None,
    units: str = "",
    vmin: float | None = None,
    vmax: float | None = None,
    symmetric: bool = False,
    n_levels: int = 20,
    dpi: int = 150,
    projection: str = "PlateCarree",
) -> None:
    """
    Save a global map of *data* on a cartopy projection (no coastlines).

    *lon*/*lat* may be 1D (regular grid, e.g. ``atm``) or 2D curvilinear
    (e.g. ``fakelnd``'s ``longitude2D``/``latitude2D``) — ``pcolormesh``
    accepts either.
    """
    values = np.asarray(data)
    cmap = cmap or DEFAULT_CMAPS.get(str(data.name), "viridis")
    vmin, vmax = _resolve_clim(values, vmin, vmax, symmetric)
    norm = _make_norm(cmap, vmin, vmax, n_levels)

    proj = get_projection(projection)
    fig, ax = plt.subplots(figsize=(10, 5), subplot_kw={"projection": proj})
    ax.set_global()

    mesh = ax.pcolormesh(
        lon, lat, values,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        norm=norm,
        shading="auto",
    )
    _add_colorbar(fig, ax, mesh, units)
    ax.set_title(title, fontsize=11)
    _save(fig, output_path, dpi)


def index_map(
    data: xr.DataArray,
    x: np.ndarray,
    y: np.ndarray,
    x_label: str,
    y_label: str,
    title: str,
    output_path: str | Path,
    cmap: str | None = None,
    units: str = "",
    vmin: float | None = None,
    vmax: float | None = None,
    symmetric: bool = False,
    n_levels: int = 20,
    dpi: int = 150,
) -> None:
    """
    Save a plain (non-georeferenced) map of *data* against index coordinates.

    Used for grids without real coordinate values (currently ``ocn`` — see
    module docstring).
    """
    values = np.asarray(data)
    cmap = cmap or DEFAULT_CMAPS.get(str(data.name), "viridis")
    vmin, vmax = _resolve_clim(values, vmin, vmax, symmetric)
    norm = _make_norm(cmap, vmin, vmax, n_levels)

    fig, ax = plt.subplots(figsize=(10, 5))
    mesh = ax.pcolormesh(x, y, values, cmap=cmap, norm=norm, shading="auto")
    _add_colorbar(fig, ax, mesh, units)
    ax.set_xlabel(f"{x_label} (index)")
    ax.set_ylabel(f"{y_label} (index)")
    ax.set_title(title, fontsize=11)
    _save(fig, output_path, dpi)


def zonal_section(
    data: xr.DataArray,
    y: np.ndarray,
    z: np.ndarray,
    y_label: str,
    z_label: str,
    title: str,
    output_path: str | Path,
    cmap: str | None = None,
    units: str = "",
    vmin: float | None = None,
    vmax: float | None = None,
    symmetric: bool = False,
    n_levels: int = 20,
    invert_yaxis: bool = False,
    dpi: int = 150,
) -> None:
    """
    Save a latitude vs. level/depth section of *data*.

    *data* is expected to have dimensions ``(z, y)`` (vertical, meridional).
    """
    values = np.asarray(data)
    cmap = cmap or DEFAULT_CMAPS.get(str(data.name), "viridis")
    vmin, vmax = _resolve_clim(values, vmin, vmax, symmetric)
    norm = _make_norm(cmap, vmin, vmax, n_levels)

    fig, ax = plt.subplots(figsize=(10, 5))
    mesh = ax.pcolormesh(y, z, values, cmap=cmap, norm=norm, shading="auto")
    if invert_yaxis:
        ax.invert_yaxis()
    ax.set_xlabel(y_label)
    ax.set_ylabel(z_label)
    _add_colorbar(fig, ax, mesh, units)
    ax.set_title(title, fontsize=11)
    _save(fig, output_path, dpi)


def overturning_streamfunction(
    psi: xr.DataArray,
    y: np.ndarray,
    z: np.ndarray,
    y_label: str,
    z_label: str,
    title: str,
    output_path: str | Path,
    units: str = "Sv",
    vmin: float | None = None,
    vmax: float | None = None,
    n_levels: int = 20,
    invert_yaxis: bool = False,
    dpi: int = 150,
) -> None:
    """
    Save a meridional overturning streamfunction section (filled contours
    plus contour lines, with the zero-transport line highlighted).

    *psi* is expected to have dimensions ``(z, y)`` (vertical, meridional),
    diverging around zero (positive/negative circulation cells).
    """
    values = np.asarray(psi)
    vmin, vmax = _resolve_clim(values, vmin, vmax, symmetric=True)
    cmap = "RdBu_r"
    norm = _make_norm(cmap, vmin, vmax, n_levels)

    fig, ax = plt.subplots(figsize=(10, 5))
    mesh = ax.pcolormesh(y, z, values, cmap=cmap, norm=norm, shading="auto")
    levels = np.linspace(vmin, vmax, n_levels + 1)
    ax.contour(y, z, values, levels=levels, colors="k", linewidths=0.4, alpha=0.6)
    ax.contour(y, z, values, levels=[0.0], colors="k", linewidths=1.2)
    if invert_yaxis:
        ax.invert_yaxis()
    ax.set_xlabel(y_label)
    ax.set_ylabel(z_label)
    _add_colorbar(fig, ax, mesh, units)
    ax.set_title(title, fontsize=11)
    _save(fig, output_path, dpi)


def meridional_transport(
    y: np.ndarray,
    values: np.ndarray,
    y_label: str,
    title: str,
    output_path: str | Path,
    units: str = "",
    dpi: int = 150,
) -> None:
    """Save a 1D meridional-transport line plot (value vs. latitude/y)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(y, np.asarray(values), linewidth=1.5)
    ax.axhline(0.0, color="k", linewidth=0.6)
    ax.set_xlabel(y_label)
    ax.set_ylabel(units or "transport")
    ax.set_title(title, fontsize=11)
    ax.grid(True, alpha=0.3)
    _save(fig, output_path, dpi)


def profile(
    data: xr.DataArray,
    z: np.ndarray,
    z_label: str,
    title: str,
    output_path: str | Path,
    units: str = "",
    dpi: int = 150,
) -> None:
    """Save a 1D vertical profile (value vs. level/depth)."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(np.asarray(data), z)
    ax.set_xlabel(units or str(data.name))
    ax.set_ylabel(z_label)
    ax.set_title(title, fontsize=11)
    ax.grid(True, alpha=0.3)
    _save(fig, output_path, dpi)


def meridional_profile(
    y: np.ndarray,
    values: np.ndarray,
    y_label: str,
    title: str,
    output_path: str | Path,
    units: str = "",
    dpi: int = 150,
) -> None:
    """Save a 1D zonal-mean line plot (value vs. latitude/y).

    Unlike :func:`meridional_transport`, there's no ``axhline(0)`` reference
    line -- that's specific to transport diagnostics (which are physically
    signed around zero) and would otherwise force fields like temperature
    or salinity to autoscale down to 0, defeating the point of a plot meant
    to show contrast within their real data range.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(y, np.asarray(values), linewidth=1.5)
    ax.set_xlabel(y_label)
    ax.set_ylabel(units or "value")
    ax.set_title(title, fontsize=11)
    ax.grid(True, alpha=0.3)
    _save(fig, output_path, dpi)


def time_series(
    times: np.ndarray,
    values: np.ndarray | dict[str, np.ndarray],
    title: str,
    ylabel: str,
    output_path: str | Path,
    dpi: int = 150,
    running_mean_steps: int | None = None,
) -> None:
    """Save a line plot of *values* against *times*.

    *values* may be a 1-D array (single line) or a dict mapping region
    names to 1-D arrays (one labelled line per region).

    When *running_mean_steps* is a positive integer N, each series is drawn
    twice: the raw data as a thin translucent line, and an N-step centred
    rolling mean as a thick opaque line of the same color.
    """
    fig, ax = plt.subplots(figsize=(10, 4))

    def _plot_one(vals: np.ndarray, label: str | None) -> None:
        if running_mean_steps:
            line, = ax.plot(times, vals, linewidth=0.8, alpha=0.35)
            rm = _rolling_mean(vals, running_mean_steps)
            ax.plot(times, rm, linewidth=2.0, color=line.get_color(), label=label)
        else:
            ax.plot(times, vals, linewidth=1.0, label=label)

    if isinstance(values, dict):
        for label, vals in values.items():
            _plot_one(vals, label)
        ax.legend(fontsize=8, loc="best")
    else:
        _plot_one(values, None)

    plot_title = title
    if running_mean_steps:
        plot_title = f"{title} (raw + {running_mean_steps}-step mean)"
    ax.set_title(plot_title, fontsize=11)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("time")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, output_path, dpi)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _make_norm(cmap: str, vmin: float, vmax: float, n_levels: int) -> mcolors.BoundaryNorm:
    bounds = np.linspace(float(vmin), float(vmax), int(n_levels) + 1)
    return mcolors.BoundaryNorm(bounds, ncolors=plt.get_cmap(cmap).N)


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Centred N-point rolling mean with edge padding to preserve array length."""
    if window <= 1:
        return values.astype(float)
    kernel = np.ones(window) / window
    half = window // 2
    padded = np.pad(values.astype(float), (half, window - 1 - half), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _add_colorbar(fig, ax, mesh, units: str) -> None:
    cbar = fig.colorbar(mesh, ax=ax, orientation="vertical", pad=0.02, shrink=0.85)
    if units:
        cbar.set_label(units, fontsize=9)
    cbar.ax.tick_params(labelsize=8)


def _resolve_clim(
    values: np.ndarray,
    vmin: float | None,
    vmax: float | None,
    symmetric: bool,
) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    vmin = None if vmin is None else float(vmin)
    vmax = None if vmax is None else float(vmax)
    if vmin is None:
        vmin = float(np.percentile(finite, 2))
    if vmax is None:
        vmax = float(np.percentile(finite, 98))
    if vmin == vmax:
        vmin, vmax = vmin - 0.5, vmax + 0.5
    if symmetric:
        bound = max(abs(vmin), abs(vmax))
        vmin, vmax = -bound, bound
    return vmin, vmax


def _save(fig: plt.Figure, output_path: str | Path, dpi: int) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
