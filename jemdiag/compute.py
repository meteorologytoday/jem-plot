"""Area-weighted reductions over the horizontal grid."""

import numpy as np
import xarray as xr

from . import grid


def latitude_weights(lat_values):
    """Return ``cos(latitude)`` weights (latitude in degrees)."""
    return np.cos(np.deg2rad(np.asarray(lat_values, dtype=float)))


def weighted_mean(da, dims):
    """Mean of ``da`` over ``dims``.

    When the meridional dimension is reduced and has real (non-synthetic)
    coordinate values, the reduction is weighted by ``cos(latitude)`` to
    approximate an area-weighted average. Otherwise a plain mean is used.
    """
    _, y_dim = grid.horizontal_dims(da)
    if y_dim in dims and not da[y_dim].attrs.get("jemdiag_synthetic", False):
        weights = xr.DataArray(
            latitude_weights(da[y_dim].values),
            dims=[y_dim],
            coords={y_dim: da[y_dim]},
        )
        return da.weighted(weights).mean(dim=dims, keep_attrs=True)
    return da.mean(dim=dims, keep_attrs=True)
