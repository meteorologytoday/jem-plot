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
3. For those same one-record-per-file runs, a record's index maps 1:1 to
   its source file's position in the sorted file list. :func:`load_all`
   uses this so it only has to open the files a run's configured
   diagnostics will actually touch (:func:`required_file_range`), rather
   than every file in the run directory -- the dominant cost for a run
   with hundreds/thousands of files, well before any array data is read
   (xarray's dask backend is already lazy about that part). The loaded
   ``time`` coordinate keeps each record's true *global* index (not
   renumbered from 0) so a restricted load is transparent to the rest of
   the package -- see :func:`jemdiag.selection.select_along`'s
   ``by_label``.
"""

from pathlib import Path

import numpy as np
import xarray as xr

from . import selection

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


def _time_spec_bound(spec):
    """Return ``(start, end)`` (``end`` exclusive) file-index bounds for one
    ``time`` spec, or ``None`` if it can't be bounded -- a bare ``"mean"``
    (mean over the *whole*, as-yet-unknown-size axis) or an open-ended
    slice like ``"500:"``.
    """
    if isinstance(spec, int):
        return spec, spec + 1
    if spec == "mean":
        return None
    if ":" in spec:
        start, end = spec.split(":")
        if not start or not end:
            return None
        return int(start), int(end)
    return int(spec), int(spec) + 1


def required_file_range(config, component, diag_types=None):
    """Return the ``(start, end)`` file-index range (``end`` exclusive)
    covering every requested diag_type's configured ``time`` selection for
    ``component``, or ``None`` if the full file range must be loaded.

    ``None`` (load everything) whenever: ``timeseries`` is configured and
    requested (it always plots the full time axis, ignoring the shared
    ``diagnostics.time`` default -- see
    :mod:`jemdiag.diagnostics.atm`'s module docstring); no diag_type is
    configured/requested for ``component``; or any requested diag_type's
    resolved ``time`` spec is unbounded (see :func:`_time_spec_bound`).
    Only relevant for the one-record-per-file runs described in this
    module's docstring -- harmless (just returns a wide/full range) for
    any other run, since a record's index no longer maps to a file
    position there anyway.
    """
    diagnostics_cfg = config.get("diagnostics", {})
    comp_cfg = diagnostics_cfg.get(component, {})
    if not isinstance(comp_cfg, dict):
        return None
    requested = (
        comp_cfg if diag_types is None
        else {name: comp_cfg[name] for name in diag_types if name in comp_cfg}
    )

    if "timeseries" in requested:
        return None

    bounds = []
    for dtype_cfg in requested.values():
        if not isinstance(dtype_cfg, dict):
            continue
        time_arg = dtype_cfg.get("time", diagnostics_cfg.get("time", "mean"))
        for spec in selection.normalize_time_args(time_arg):
            bound = _time_spec_bound(spec)
            if bound is None:
                return None
            bounds.append(bound)

    if not bounds:
        return None
    starts, ends = zip(*bounds)
    return max(0, min(starts)), max(ends)


def load_component(run_dir, component, file_range=None):
    """Load files for ``component`` into a single, lazily-backed Dataset.

    ``file_range``, if given, is an ``(start, end)`` file-index range (as
    from :func:`required_file_range`) -- only files in that range are
    opened, and the resulting ``time`` coordinate is offset by ``start``
    so it keeps carrying each record's true global index rather than
    being renumbered from 0 (see this module's docstring).
    """
    files = discover_files(run_dir, component)
    offset = 0
    if file_range is not None:
        start, end = file_range
        files = files[start:end]
        if not files:
            raise FileNotFoundError(
                f"No '{component}-*.nc' files in index range {file_range} "
                f"in {run_dir}"
            )
        offset = start
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
    if offset:
        ds = ds.assign_coords(time=ds["time"] + offset)
    return ds


def load_all(config, diag_types=None):
    """Load every jem component referenced by ``config["data"]["run_dir"]``.

    Components whose files are absent from the run directory are silently
    skipped (mirrors how a diagnostics run skips components with no data),
    so a run directory only containing a subset of ``atm``/``ocn``/
    ``fakelnd`` still loads successfully.

    ``diag_types``, if given, restricts loading per component to the file
    range :func:`required_file_range` computes for those diag_types --
    see its docstring for when that's possible vs. when the full range is
    loaded.
    """
    run_dir = config["data"]["run_dir"]
    result = {}
    for component in COMPONENTS:
        try:
            file_range = required_file_range(config, component, diag_types)
            result[component] = load_component(run_dir, component, file_range=file_range)
        except FileNotFoundError:
            continue
    if not result:
        raise FileNotFoundError(
            f"No jem component files found in {run_dir}"
        )
    return result
