"""Discovering and loading jem output files.

A jem run directory contains one or more numbered NetCDF files per
component, e.g. ``atm-00000.nc``, ``atm-00001.nc``, .... These are opened
lazily with :func:`xarray.open_mfdataset` and concatenated along ``time``.
Two quirks of the raw output are normalized here so the rest of the package
can treat every component the same way:

1. ``ocn`` files have no coordinate values for any dimension. Every
   dimension that lacks a coordinate is given a synthetic integer-index
   coordinate (0, 1, 2, ...), tagged with ``jemdiag_synthetic=True``. Plot
   code (see :mod:`jemdiag.plot_utils` and :mod:`jemdiag.diagnostics.ocn`)
   checks this tag to fall back to an index-based map instead of a cartopy
   projection.
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


def _ensure_index_coords(ds):
    """Add a synthetic integer-index coordinate to any dimension without one."""
    new_coords = {}
    for dim, size in ds.sizes.items():
        if dim not in ds.coords:
            new_coords[dim] = xr.Variable(
                dim, np.arange(size), attrs={"jemdiag_synthetic": True}
            )
    if new_coords:
        ds = ds.assign_coords(new_coords)
    return ds


def _normalize_time(ds):
    """Replace a missing/degenerate ``time`` coordinate with a record index."""
    if "time" not in ds.dims:
        return ds
    if "time" not in ds.coords:
        return ds.assign_coords(time=("time", np.arange(ds.sizes["time"])))

    index = ds.indexes["time"]
    degenerate = len(index) > 1 and (
        not index.is_unique or not index.is_monotonic_increasing
    )
    if degenerate:
        ds = ds.drop_vars("time")
        ds = ds.assign_coords(time=("time", np.arange(ds.sizes["time"])))
        ds["time"].attrs["note"] = (
            "synthetic record index (original time coordinate was "
            "missing, duplicated, or non-monotonic)"
        )
    return ds


def load_component(run_dir, component):
    """Load all files for ``component`` into a single, lazily-backed Dataset."""
    files = discover_files(run_dir, component)
    ds = xr.open_mfdataset(
        files,
        combine="nested",
        concat_dim="time",
        data_vars="minimal",
        coords="minimal",
        compat="override",
    )
    ds = _ensure_index_coords(ds)
    ds = _normalize_time(ds)
    return ds


def load_all(config):
    """Load every jem component referenced by ``config["data"]["run_dir"]``.

    Components whose files are absent from the run directory are silently
    skipped (mirrors how a diagnostics run skips components with no data),
    so a run directory only containing a subset of ``atm``/``ocn``/
    ``fakelnd`` still loads successfully.
    """
    run_dir = config["data"]["run_dir"]
    result = {}
    for component in COMPONENTS:
        try:
            result[component] = load_component(run_dir, component)
        except FileNotFoundError:
            continue
    if not result:
        raise FileNotFoundError(
            f"No jem component files found in {run_dir}"
        )
    return result
