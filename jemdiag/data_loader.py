"""Discovering and loading jem output files.

A jem run directory contains one or more numbered NetCDF files per
component (:data:`COMPONENTS`), e.g. ``atm-00000000.nc``,
``atm-00000073.nc``, .... These are opened lazily with
:func:`xarray.open_mfdataset` and concatenated along ``time``. The raw
output is normalized here so the rest of the package can treat every
component and every jem version the same way:

1. ``ocn`` files have no coordinate values for any dimension. Every
   dimension that lacks a coordinate is given a synthetic integer-index
   coordinate (0, 1, 2, ...), tagged with ``jemdiag_synthetic=True``. Plot
   code (see :mod:`jemdiag.plot_utils` and :mod:`jemdiag.diagnostics.ocn`)
   checks this tag to fall back to an index-based map instead of a cartopy
   projection.
2. The ``time`` coordinate is *always* replaced by the record's integer
   index in the run (0, 1, 2, ...); the raw values are never decoded.
   What jem writes there has changed between versions: older runs wrote a
   single non-informative value per file, newer ones write real dates
   that run past year 2262 for long runs, where numpy's ``datetime64``
   overflows and xarray silently switches to ``cftime`` objects partway
   through a run. A record index behaves identically for all of them,
   and it is what every config ``time`` spec (``"4990:5000"``, ...) means.
3. Where the model calendar is known, each record also gets a
   ``model_year`` coordinate along ``time`` (the chunk's midpoint, in
   365-day years since the start of the run) for labelling time axes --
   see :func:`_model_years`. It is derived from the start day jem encodes
   in each file name (``ocn-00000073.nc`` starts on day 73) and the chunk
   length ``simulation_interval_days`` in the run's ``run_args.json``; a
   run missing either simply has no ``model_year``.
4. Most runs write one record per file, so a record's index maps 1:1 to
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

import json
import re
from pathlib import Path

import numpy as np
import xarray as xr

from . import selection

COMPONENTS = ("atm", "ocn", "lnd", "seaice")

# Length of a model year in days. jem runs use a 365-day year: a 73-day
# chunk is exactly a fifth of one, and run_args.json's
# total_simulation_days is a multiple of 365.
DAYS_PER_YEAR = 365

RUN_ARGS_FILE = "run_args.json"


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


def read_run_args(run_dir):
    """Return the run's ``run_args.json`` as a dict, or ``{}`` if absent.

    jem's driver writes this file next to the output; older runs predate
    it. A file that exists but isn't a JSON object raises ``ValueError``.
    """
    path = Path(run_dir) / RUN_ARGS_FILE
    if not path.is_file():
        return {}
    with path.open() as fh:
        args = json.load(fh)
    if not isinstance(args, dict):
        raise ValueError(f"{path} must hold a JSON object, got {type(args).__name__}")
    return args


def _file_start_day(path, component):
    """Return the start day jem encodes in a file name, or ``None``.

    ``ocn-00000073.nc`` -> ``73``. Anchored on the component prefix rather
    than fixed character offsets, so it works for any component name.
    """
    match = re.fullmatch(rf"{re.escape(component)}-(\d+)\.nc", Path(path).name)
    return int(match.group(1)) if match else None


def _model_years(files, component, run_args):
    """Return each file's chunk-midpoint model year, or ``None`` if unknown.

    Needs ``simulation_interval_days`` from ``run_args.json`` (the chunk
    length) and a start day in every file name; returns ``None`` when
    either is missing rather than guessing -- older runs number their
    files by record, not by day, and the two are indistinguishable from
    the name alone.
    """
    interval = run_args.get("simulation_interval_days")
    if interval is None:
        return None
    start_days = [_file_start_day(f, component) for f in files]
    if any(day is None for day in start_days):
        return None
    return (np.asarray(start_days, dtype=float) + interval / 2) / DAYS_PER_YEAR


def _index_time(ds, offset):
    """Replace ``time`` with the integer record index, starting at *offset*."""
    if "time" not in ds.dims:
        return ds
    index = np.arange(ds.sizes["time"]) + offset
    ds = ds.drop_vars("time", errors="ignore")
    ds = ds.assign_coords(time=("time", index))
    ds["time"].attrs["long_name"] = "record index in the run"
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
    restricted = file_range is not None
    if restricted:
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
        decode_times=False,  # replaced by a record index below; see module docstring
    )
    one_record_per_file = "time" in ds.dims and ds.sizes["time"] == len(files)
    if restricted and not one_record_per_file:
        raise ValueError(
            f"Cannot load a file-index range for '{component}' in {run_dir}: "
            f"its files don't hold exactly one record each, so a record "
            f"index doesn't map to a file position"
        )
    ds = _ensure_index_coords(ds)
    ds = _index_time(ds, offset)
    if one_record_per_file:
        years = _model_years(files, component, read_run_args(run_dir))
        if years is not None:
            ds = ds.assign_coords(model_year=("time", years))
            ds["model_year"].attrs = {
                "long_name": "model year (chunk midpoint)",
                "units": f"{DAYS_PER_YEAR}-day years",
            }
    return ds


def load_all(config, diag_types=None, components=None):
    """Load every jem component referenced by ``config["data"]["run_dir"]``.

    Components whose files are absent from the run directory are silently
    skipped (mirrors how a diagnostics run skips components with no data),
    so a run directory only containing a subset of :data:`COMPONENTS`
    still loads successfully.

    ``components``, if given, restricts loading to those components (each
    must be in :data:`COMPONENTS`) -- opening a few thousand files is the
    dominant cost, so components that won't be plotted aren't touched.

    ``diag_types``, if given, restricts loading per component to the file
    range :func:`required_file_range` computes for those diag_types --
    see its docstring for when that's possible vs. when the full range is
    loaded.
    """
    run_dir = config["data"]["run_dir"]
    if components is None:
        components = COMPONENTS
    unknown = sorted(set(components) - set(COMPONENTS))
    if unknown:
        raise ValueError(f"Unknown component(s) {unknown}; choose from {list(COMPONENTS)}")
    result = {}
    for component in components:
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
