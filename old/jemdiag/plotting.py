"""Matplotlib plotting functions for jem diagnostics.

Each function takes an already-reduced :class:`xarray.DataArray` (see
:mod:`jemdiag.selection`) and returns ``(fig, ax)``, so it can be used
standalone (e.g. from a notebook) as well as from :mod:`plot_jem`.
"""

import matplotlib.pyplot as plt

from . import grid


def _get_fig_ax(ax):
    if ax is None:
        return plt.subplots()
    return ax.figure, ax


def _colorbar_label(da):
    units = da.attrs.get("units")
    name = da.attrs.get("long_name") or da.attrs.get("description") or da.name
    return f"{name} ({units})" if units else str(name)


def plot_map(da2d, ax=None, cmap=None, vmin=None, vmax=None, title=None):
    """Plot a 2D horizontal (lat-lon) map with ``pcolormesh``."""
    x_dim, y_dim = grid.horizontal_dims(da2d)
    x, y, _ = grid.horizontal_coords(da2d, x_dim, y_dim)
    da2d = da2d.transpose(y_dim, x_dim)

    fig, ax = _get_fig_ax(ax)
    mesh = ax.pcolormesh(x, y, da2d.values, cmap=cmap, vmin=vmin, vmax=vmax, shading="auto")
    fig.colorbar(mesh, ax=ax, label=_colorbar_label(da2d))
    ax.set_xlabel(x_dim)
    ax.set_ylabel(y_dim)
    ax.set_title(title or str(da2d.name))
    return fig, ax


def plot_timeseries(da1d, ax=None, title=None):
    """Plot a 1D time series."""
    fig, ax = _get_fig_ax(ax)
    ax.plot(da1d["time"].values, da1d.values)
    ax.set_xlabel(
        "time"
        if "note" not in da1d["time"].attrs
        else "time (synthetic record index)"
    )
    ax.set_ylabel(_colorbar_label(da1d))
    ax.set_title(title or str(da1d.name))
    fig.autofmt_xdate()
    return fig, ax


def plot_zonal_mean(da2d, ax=None, cmap=None, vmin=None, vmax=None, invert_yaxis=False, title=None):
    """Plot a zonal-mean cross section (meridional dim vs. level/depth)."""
    z_dim = grid.vertical_dim(da2d)
    if z_dim is None:
        raise ValueError(
            "plot_zonal_mean requires a DataArray with a vertical "
            "dimension (level/depth)"
        )
    y_dim = [d for d in da2d.dims if d != z_dim][0]
    da2d = da2d.transpose(z_dim, y_dim)

    fig, ax = _get_fig_ax(ax)
    cf = ax.contourf(da2d[y_dim].values, da2d[z_dim].values, da2d.values, cmap=cmap, vmin=vmin, vmax=vmax)
    fig.colorbar(cf, ax=ax, label=_colorbar_label(da2d))
    ax.set_xlabel(y_dim)
    ax.set_ylabel(z_dim)
    if invert_yaxis:
        ax.invert_yaxis()
    ax.set_title(title or str(da2d.name))
    return fig, ax


def plot_profile(da1d, ax=None, title=None):
    """Plot a 1D vertical profile (value vs. level/depth)."""
    z_dim = grid.vertical_dim(da1d)
    if z_dim is None:
        raise ValueError(
            "plot_profile requires a DataArray with a vertical "
            "dimension (level/depth)"
        )

    fig, ax = _get_fig_ax(ax)
    ax.plot(da1d.values, da1d[z_dim].values)
    ax.set_xlabel(_colorbar_label(da1d))
    ax.set_ylabel(z_dim)
    ax.set_title(title or str(da1d.name))
    return fig, ax
