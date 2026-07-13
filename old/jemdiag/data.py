"""Discovering and loading jem output files.

A jem run directory contains one or more numbered NetCDF files per
component, e.g. ``atm-00000.nc``, ``atm-00001.nc``, .... These are opened
lazily and concatenated along ``time``. Two quirks of the raw output are
normalized here so the rest of the package can treat every component the
same way:

1. ``ocn`` files have no coordinate values for any dimension. Every
   dimension that lacks a coordinate is given a synthetic integer-index
   coordinate (0, 1, 2, ...), tagged with ``jemdiag_synthetic=True``.
2. Some runs (e.g. long, low-frequency-output runs) write a single,
   non-informative ``time`` value per file. After concatenation this shows
   up as a missing, duplicated, or non-monotonic ``time`` coordinate, which
   is replaced with a synthetic record index (0, 1, 2, ...).
"""

from pathlib import Path

import numpy as np
import xarray as xr

COMPONENTS = ("atm", "ocn", "fakelnd")


def discover_files(run_dir, component):
    """Return the sorted list of ``{component}-*.nc`` files in ``run_dir``."""
    run_dir = Path(run_dir)
    files = sorted(run_dir.glob(f"{component}-*.nc"))
    if not files:
        raise FileNotFoundError(
            f"No '{component}-*.nc' files found in {run_dir}"
        )
    return files


def list_variables(run_dir, component):
    """Return the sorted list of data variable names for ``component``."""
    files = discover_files(run_dir, component)
    with xr.open_dataset(files[0]) as ds:
        return sorted(ds.data_vars)


def _ensure_index_coords(da):
    """Add a synthetic integer-index coordinate to any dimension without one."""
    new_coords = {}
    for dim in da.dims:
        if dim not in da.coords:
            new_coords[dim] = xr.Variable(
                dim, np.arange(da.sizes[dim]), attrs={"jemdiag_synthetic": True}
            )
    if new_coords:
        da = da.assign_coords(new_coords)
    return da


def _normalize_time(da):
    """Replace a missing/degenerate ``time`` coordinate with a record index."""
    if "time" not in da.dims:
        return da
    if "time" not in da.coords:
        return da.assign_coords(time=("time", np.arange(da.sizes["time"])))

    index = da.indexes["time"]
    degenerate = len(index) > 1 and (
        not index.is_unique or not index.is_monotonic_increasing
    )
    if degenerate:
        da = da.drop_vars("time")
        da = da.assign_coords(time=("time", np.arange(da.sizes["time"])))
        da["time"].attrs["note"] = (
            "synthetic record index (original time coordinate was "
            "missing, duplicated, or non-monotonic)"
        )
    return da


def load_variable(run_dir, component, varname):
    """Load a single variable for ``component``, concatenated across files."""
    files = discover_files(run_dir, component)
    pieces = []
    for path in files:
        ds = xr.open_dataset(path)
        if varname not in ds.data_vars:
            available = ", ".join(sorted(ds.data_vars))
            raise KeyError(
                f"Variable '{varname}' not found in {path.name}. "
                f"Available variables: {available}"
            )
        pieces.append(_ensure_index_coords(ds[varname]))

    combined = pieces[0] if len(pieces) == 1 else xr.concat(pieces, dim="time")
    return _normalize_time(combined)
