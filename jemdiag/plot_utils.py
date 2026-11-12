"""
Shared plotting utilities for jem diagnostic figures.

Horizontal maps are produced with cartopy (for a correct geographic
projection) whenever the underlying grid has real coordinates (``atm``,
``fakelnd``). Earth coastlines are off by default (jem runs are idealized
unless configured with real topography) and opt-in per experiment via the
config ``coastlines`` toggle (see :func:`resolve_coastlines`). ``ocn``
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
import cmocean  # noqa: F401 -- registers the "cmo.*" colormaps with matplotlib on import
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
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
    "precipitation": "cmo.rain",
    # ocean
    "temp": "cmo.thermal",
    "salt": "PuBu",
    "sea_surface_temperature": "cmo.thermal",
    "sea_surface_salinity": "PuBu",
    "heat_flux": "RdBu_r",
    "freshwater_flux": "BrBG",
    # fakelnd
    "mixed_layer_depth": "viridis",
    "total_heat_flux": "RdBu_r",
}


# Default cartopy styling for the opt-in `coastlines` map option. Every key
# here is also the exact set of style overrides a config mapping may set
# (see resolve_coastlines).
COASTLINE_DEFAULTS: dict = {
    "resolution": "110m",
    "color": "black",
    "linewidth": 0.6,
}

COASTLINE_RESOLUTIONS = ("110m", "50m", "10m")


# ── Config helpers ────────────────────────────────────────────────────────────

def _resolve_map_option(key: str, diag_cfg: dict, diagnostics_cfg: dict, default):
    """Resolve one map option from a diag_type block, falling back to the
    shared ``diagnostics.<key>`` default and finally to *default*.

    Same precedence as ``time`` (see
    :func:`jemdiag.selection.resolve_time_spec`), so an experiment can set an
    option once for every component while a single diag_type can still opt
    out with its own value.
    """
    return diag_cfg.get(key, diagnostics_cfg.get(key, default))


def resolve_central_longitude(diag_cfg: dict, diagnostics_cfg: dict) -> float:
    """Resolve the ``central_longitude`` map option (degrees east).

    Recenters the cartopy projection, e.g. ``200`` puts 200E in the middle
    of the figure instead of the 0 default -- useful for keeping the Pacific
    from being split across the two edges. Raises ``ValueError`` for a
    non-numeric value.
    """
    value = _resolve_map_option("central_longitude", diag_cfg, diagnostics_cfg, 0.0)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"`central_longitude` must be a number in degrees east, got {value!r}"
        )
    return float(value)


def resolve_coastlines(diag_cfg: dict, diagnostics_cfg: dict) -> dict | None:
    """Resolve the ``coastlines`` map option into ``ax.coastlines()`` kwargs.

    Falls back, in order, to a diag_type-level ``coastlines`` override, then
    the shared ``diagnostics.coastlines`` default, then off -- the same
    precedence as ``time`` (see :func:`jemdiag.selection.resolve_time_spec`).
    Off is the default because jem runs are idealized unless configured with
    real Earth topography, in which case real coastlines would be misleading.

    The config value may be a bool, or a mapping of style overrides
    (``resolution``/``color``/``linewidth``, see :data:`COASTLINE_DEFAULTS`)
    which implies "on". Returns the kwargs to pass to ``Axes.coastlines()``,
    or ``None`` when disabled. Raises ``ValueError`` on an unusable value so
    a typo surfaces immediately instead of silently drawing nothing.
    """
    spec = _resolve_map_option("coastlines", diag_cfg, diagnostics_cfg, False)
    if spec is None or spec is False:
        return None
    if spec is True:
        return dict(COASTLINE_DEFAULTS)
    if not isinstance(spec, dict):
        raise ValueError(
            f"`coastlines` must be true/false or a mapping of "
            f"{sorted(COASTLINE_DEFAULTS)}, got {spec!r}"
        )
    unknown = sorted(set(spec) - set(COASTLINE_DEFAULTS))
    if unknown:
        raise ValueError(
            f"Unknown `coastlines` option(s) {unknown}. "
            f"Choose from: {sorted(COASTLINE_DEFAULTS)}"
        )
    options = {**COASTLINE_DEFAULTS, **spec}
    if options["resolution"] not in COASTLINE_RESOLUTIONS:
        raise ValueError(
            f"Unknown `coastlines.resolution` {options['resolution']!r}. "
            f"Choose from: {list(COASTLINE_RESOLUTIONS)}"
        )
    return options


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
    central_longitude: float = 0.0,
    coastlines: dict | None = None,
) -> None:
    """
    Save a global map of *data* on a cartopy projection.

    *lon*/*lat* may be 1D (regular grid, e.g. ``atm``) or 2D curvilinear
    (e.g. ``fakelnd``'s ``longitude2D``/``latitude2D``) — ``pcolormesh``
    accepts either.

    *central_longitude* (degrees east) recenters the projection; the data
    itself is unchanged, cartopy wraps the cells that straddle the resulting
    map edge.

    *coastlines* are ``ax.coastlines()`` kwargs from
    :func:`resolve_coastlines`, or ``None`` (the default) to draw none.
    """
    values = np.asarray(data)
    cmap = cmap or DEFAULT_CMAPS.get(str(data.name), "viridis")
    vmin, vmax = _resolve_clim(values, vmin, vmax, symmetric)
    norm = _make_norm(cmap, vmin, vmax, n_levels)

    proj = get_projection(projection, central_longitude=central_longitude)
    fig, ax = plt.subplots(figsize=(10, 5), subplot_kw={"projection": proj})
    ax.set_global()

    mesh = ax.pcolormesh(
        lon, lat, values,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        norm=norm,
        shading="auto",
    )
    if coastlines:
        # Explicit zorder keeps the outline above the mesh (zorder 1) rather
        # than relying on cartopy's default feature ordering.
        ax.coastlines(zorder=2.5, **coastlines)
    gl = ax.gridlines(
        crs=ccrs.PlateCarree(), draw_labels=True,
        linewidth=0.4, color="gray", alpha=0.5, linestyle="--",
    )
    gl.top_labels = False
    gl.right_labels = False
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
    ax.grid()
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


def _clean_number(value: float, _pos=None) -> str:
    """Format a tick value as an integer when it's whole, else ``%.1f``.

    Colorbar tick values come from :func:`_make_norm`'s evenly-spaced
    ``BoundaryNorm`` levels, which are rarely round numbers (e.g.
    ``18.37...``) even when the data range itself is -- this keeps labels
    readable instead of matplotlib's default (often 4+ significant digits).
    """
    label = f"{value:.0f}" if np.isclose(value, round(value)) else f"{value:.1f}"
    return "0" if label == "-0" else "0.0" if label == "-0.0" else label


def _colorbar_scale_exponent(vmin: float, vmax: float) -> int | None:
    """Return an exponent ``E`` such that dividing tick values by ``10**E``
    keeps them readable under :func:`_clean_number`'s 1-decimal cap, or
    ``None`` if no scaling is needed.

    Triggered only when the whole displayed range is under 1 in magnitude
    (e.g. precipitation, ~0-0.15 in jem's raw units) -- left alone,
    ``%.1f`` collapses several distinct ticks to the same repeated label
    (``0.1``, ``0.1``, ``0.0``, ``0.0``, ...). Follows matplotlib's own
    scientific-notation convention: the colorbar label gets a ``(x10^E)``
    suffix, and the true value is ``displayed_tick * 10**E``.
    """
    max_abs = max(abs(vmin), abs(vmax))
    if max_abs == 0 or max_abs >= 1:
        return None
    return int(np.floor(np.log10(max_abs)))


def _add_colorbar(fig, ax, mesh, units: str) -> None:
    cbar = fig.colorbar(mesh, ax=ax, orientation="vertical", pad=0.02, shrink=0.85)
    exponent = _colorbar_scale_exponent(mesh.norm.vmin, mesh.norm.vmax)
    if exponent is None:
        cbar.ax.yaxis.set_major_formatter(mticker.FuncFormatter(_clean_number))
        label = units
    else:
        scale = 10.0**exponent
        cbar.ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _pos=None: _clean_number(v / scale))
        )
        label = f"{units} (" + r"$\times10^{%d}$)" % exponent if units else r"$\times10^{%d}$" % exponent
    if label:
        cbar.set_label(label, fontsize=9)
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
