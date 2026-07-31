"""Time / level / region selection and reduction shared by all diagnostics.

Each ``reduce_to_*`` function turns a raw (possibly 3D or 4D) DataArray into
the shape expected by the corresponding plotting function in
:mod:`jemdiag.plot_utils`.
"""

from . import compute, grid


def parse_index_or_mean(arg):
    """Parse a ``"mean"`` | ``<int>`` | ``"<start>:<end>"`` argument.

    ``arg`` may already be an ``int`` (as loaded from YAML), a ``str``
    ``"mean"``, an integer-looking string, or a ``"start:end"`` slice
    string.
    """
    if isinstance(arg, int):
        return arg
    if arg == "mean":
        return "mean"
    if ":" in arg:
        start, end = arg.split(":")
        return slice(int(start) if start else None, int(end) if end else None)
    return int(arg)


def normalize_time_args(arg):
    """Normalize a ``time`` config value to a list of individual specs.

    A scalar (``"mean"`` | ``<int>`` | ``"<start>:<end>"``) is wrapped in a
    single-element list; an explicit list of such specs is returned
    unchanged. Callers loop over the result so a diag_type can produce one
    plot per requested time snapshot instead of just one plot overall.
    """
    if isinstance(arg, list):
        return arg
    return [arg]


def resolve_time_spec(diag_cfg: dict, diagnostics_cfg: dict, default="mean"):
    """Return the normalized ``time`` spec list for one diag_type block.

    Falls back, in order, to a diag_type-level ``time`` override, then the
    shared ``diagnostics.time`` default, then ``default``. This lets a whole
    experiment share one time selection across map_2d/zonal_section/profile/
    overturning_streamfunction/heat_transport while still allowing a single
    diag_type to opt out with its own explicit ``time``.
    """
    time_arg = diag_cfg.get("time", diagnostics_cfg.get("time", default))
    return normalize_time_args(time_arg)


def time_label(arg) -> str:
    """Human/filename-friendly label for a single time spec.

    Used to disambiguate output files and titles when a diag_type's
    ``time`` config lists more than one spec, e.g. "t0", "t5-10", "mean".
    """
    if isinstance(arg, int):
        return f"t{arg}"
    if arg == "mean":
        return "mean"
    if ":" in arg:
        start, end = arg.split(":")
        return f"t{start or 'start'}-{end or 'end'}"
    return f"t{int(arg)}"


def select_along(da, dim, arg):
    """Select or reduce ``da`` along ``dim`` according to ``arg``.

    If ``dim`` is ``None`` or not present in ``da``, ``da`` is returned
    unchanged (so callers don't need to special-case variables that lack a
    given dimension, e.g. a vertical level).
    """
    if dim is None or dim not in da.dims:
        return da
    parsed = parse_index_or_mean(arg)
    if parsed == "mean":
        return da.mean(dim=dim, keep_attrs=True)
    if isinstance(parsed, slice):
        return da.isel({dim: parsed}).mean(dim=dim, keep_attrs=True)
    return da.isel({dim: parsed})


def select_region(da, region):
    """Select a ``(lon_min, lon_max, lat_min, lat_max)`` bounding box.

    Returns ``da`` unchanged if ``region`` is ``None``. Raises ``ValueError``
    if the horizontal coordinates are synthetic (index-based), since a
    lon/lat bounding box is meaningless in that case.
    """
    if region is None:
        return da
    lon_min, lon_max, lat_min, lat_max = region
    x_dim, y_dim = grid.horizontal_dims(da)
    if da[x_dim].attrs.get("jemdiag_synthetic") or da[y_dim].attrs.get(
        "jemdiag_synthetic"
    ):
        raise ValueError(
            f"region selection requires real lon/lat coordinates, but "
            f"'{x_dim}'/'{y_dim}' are index-based for this component"
        )
    return da.sel({x_dim: slice(lon_min, lon_max), y_dim: slice(lat_min, lat_max)})


def reduce_to_map(da, time_arg, level_arg):
    """Reduce to a 2D ``(y, x)`` field for :func:`jemdiag.plot_utils.global_map`."""
    da = select_along(da, "time", time_arg)
    da = select_along(da, grid.vertical_dim(da), level_arg)
    return da


def reduce_to_timeseries(da, level_arg, region=None):
    """Reduce to a 1D ``(time,)`` series for :func:`jemdiag.plot_utils.time_series`."""
    da = select_along(da, grid.vertical_dim(da), level_arg)
    da = select_region(da, region)
    x_dim, y_dim = grid.horizontal_dims(da)
    return compute.weighted_mean(da, dims=[x_dim, y_dim])


def reduce_to_zonal(da, time_arg):
    """Reduce to a 2D ``(level/depth, y)`` field for :func:`jemdiag.plot_utils.zonal_section`."""
    if grid.vertical_dim(da) is None:
        raise ValueError(
            "Zonal-mean plots require a variable with a vertical "
            "dimension (level/depth)"
        )
    da = select_along(da, "time", time_arg)
    x_dim, _ = grid.horizontal_dims(da)
    return da.mean(dim=x_dim, keep_attrs=True)


def reduce_to_profile(da, time_arg, region=None):
    """Reduce to a 1D ``(level/depth,)`` profile for :func:`jemdiag.plot_utils.profile`."""
    if grid.vertical_dim(da) is None:
        raise ValueError(
            "Profile plots require a variable with a vertical "
            "dimension (level/depth)"
        )
    da = select_along(da, "time", time_arg)
    da = select_region(da, region)
    x_dim, y_dim = grid.horizontal_dims(da)
    return compute.weighted_mean(da, dims=[x_dim, y_dim])
