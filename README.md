# diagnostic-package-for-jem

Config-driven diagnostic plotting for output produced by `jax-esm` (jem),
modeled on the `plotting_poem` package: one YAML config describes an
experiment's data path, grids, and which variables to plot for each
component/diagnostic-type, and a single command produces the whole figure
set.

## Layout

```
jemdiag/                  # reusable plotting library
├── data_loader.py         # discover/load a component's *.nc files into one xr.Dataset
├── grid.py                # dimension/coordinate detection (atm/ocn/fakelnd), real vs. synthetic coords
├── compute.py              # area-weighted reductions (numpy)
├── selection.py             # time/level/region selection & reduction
├── plot_utils.py             # cartopy-based plot functions (map, zonal_section, profile, time_series)
└── diagnostics/               # one module per component (atm, ocn, fakelnd), HANDLERS registry + run()
config/
└── long_run.yaml          # experiment config for data/long_run
plot_jem.py                # CLI entry point (`run` and `list-vars` subcommands)
scripts/
└── run_diagnostic.sh       # bash wrapper around plot_jem.py
requirements.txt           # numpy, xarray, netCDF4, matplotlib, cartopy, pyyaml
```

A run directory contains numbered NetCDF files per component, e.g.
`atm-00000.nc`, `ocn-00000.nc`, `fakelnd-00000.nc`, .... `data_loader`
concatenates these along `time` automatically.

Notes on grids:
- `atm` has real `lon`/`lat`/`level` coordinates — maps use a cartopy
  `PlateCarree` projection (coastlines off unless the config's `coastlines`
  toggle turns them on, see below).
- `ocn` has real `lon`/`lat` coordinates but no `depth` coordinate values
  (treated as a plain integer index). Maps automatically use cartopy when a
  variable's horizontal coordinates are real, and fall back to a plain
  index-based map otherwise (see `jemdiag/grid.py:has_real_horizontal_coords`)
  — no config change needed if this changes in a future jem version.
  The `depth` index runs from the seafloor (index 0, thickest layers) to
  the surface (last index, thinnest layers) — `zonal_section` and
  `overturning_streamfunction` default to `invert_yaxis: false` so the
  surface is drawn at the top of the chart.
- `fakelnd` has real `time` plus 2D curvilinear `longitude2D`/`latitude2D`;
  maps use cartopy the same way as `atm`.
- If a run's `time` coordinate is missing or degenerate (e.g. one
  non-informative value per file, as in `data/long_run`), it is replaced
  with a synthetic record index (0, 1, 2, ...).

## Usage

Generate every figure the config describes:

```sh
python plot_jem.py run --config config/long_run.yaml
```

Restrict to specific components / diagnostic types (fast dev iteration):

```sh
python plot_jem.py run --config config/long_run.yaml \
    --components atm ocn --diag-types map_2d

python plot_jem.py run --config config/long_run.yaml \
    --components atm --diag-types map_2d --vars temperature
```

Point at a different run directory / experiment name without editing the
config file:

```sh
python plot_jem.py run --config config/long_run.yaml \
    --run-dir /path/to/other/run --exp-name my_run
```

List available variables for a component (does not require a config file):

```sh
python plot_jem.py list-vars --run-dir data/long_run --component atm
```

Figures are written to
`{output.base_dir}/{experiment.name}/{component}/{diag_type}/{variable}.png`
(dots in variable names become underscores), e.g.
`figures/long_run/atm/map_2d/shortwave_rad_rsds.png`. If a diag_type's `time`
resolves to more than one spec (see below), each one gets its own file with a
`_<time_label>` suffix instead, e.g. `..._t0.png`/`..._mean.png`.

## Config file schema

See `config/long_run.yaml` for a complete example. Top-level sections:

- `experiment.name` — used in output paths.
- `data.run_dir` — directory containing `{component}-*.nc` files.
- `grids.<component>.projection` — cartopy projection name (currently only
  `PlateCarree` is implemented); horizontal/vertical dimension names and
  real-vs-synthetic coordinates are auto-detected per variable, so `ocn`
  needs no grid entry at all.
- `diagnostics.time` — shared default `time` selection for every
  non-timeseries diag_type below (`map_2d`/`zonal_section`/`profile`/
  `overturning_streamfunction`/`heat_transport`); `timeseries` always plots
  the full time axis and ignores it. Any diag_type block may set its own
  `time` to override this default just for itself.
- `diagnostics.coastlines` — shared default for whether `map_2d` figures
  draw Earth coastlines; `false` (off) if omitted, since an idealized run's
  land-sea mask has nothing to do with real geography. May be `true`/`false`
  or a mapping of style overrides (`resolution`: `110m`/`50m`/`10m`,
  `color`, `linewidth`), e.g.
  `coastlines: {resolution: 50m, linewidth: 0.4}`. Any `map_2d` block may
  set its own `coastlines` to override this default just for itself. Only
  applies to cartopy maps: `ocn` falls back to an index-based map when its
  coordinates aren't real, and prints a note that the toggle was ignored.
- `diagnostics.central_longitude` — shared default for the longitude
  (degrees east) at the center of `map_2d` figures; `0` if omitted. Set e.g.
  `200` to keep the Pacific whole instead of split across the two edges.
  Only the projection is recentered — the data is untouched, and cartopy
  wraps the grid cells that straddle the new map edge. Same override rules
  as `coastlines` (per-`map_2d` override, ignored on index-based `ocn`
  maps).
- `diagnostics.<component>.<diag_type>` — one block per
  `map_2d`/`zonal_section`/`profile`/`timeseries`/`overturning_streamfunction`/
  `heat_transport` (components only produce the diag_types listed in their
  `HANDLERS`, see `jemdiag/diagnostics/`):
  - `vars`: list of variable names to plot. Not used by
    `overturning_streamfunction`/`heat_transport`, which are single derived
    quantities computed from fixed inputs (`v`/`dzt`/`mask_T` for
    `ocn.overturning_streamfunction`; `v`/`temp`/`dzt`/`mask_T` for
    `ocn.heat_transport`; `v_wind`/`temperature`/`normalized_surface_pressure`
    for `atm.heat_transport`) rather than a user-selected variable.
  - `time` (all diag_types except `timeseries`) / `level`: `"mean"`, an
    integer index, or a `"start:end"` slice (averaged over the slice). jem
    runs have no calendar, so this replaces season/month aggregation — it
    always operates on raw record index. `time` may also be a list of any
    of those, e.g. `time: [0, 12, mean]`, to produce one plot per entry
    instead of one plot overall (see the output path note above); `level`
    only ever takes a single spec.
  - `region` (`profile`/`timeseries` only): `[lon_min, lon_max, lat_min,
    lat_max]` bounding box; requires real lon/lat coordinates.
  - `lon_bands` (`ocn.overturning_streamfunction`/`ocn.heat_transport`
    only): restrict the zonal integral to a longitude sub-range
    `[lon_min, lon_max]` (e.g. an ocean basin), instead of the full globe.
    May also be a list of such pairs, e.g. `[[280, 360], [0, 20]]`, to
    produce one plot per band (filename/title suffixed accordingly, same
    pattern as a `time` list); omit for the default global integral.
    Doesn't wrap across the 0/360 boundary. Requires real lon coordinates.
  - `invert_yaxis` (`zonal_section`/`overturning_streamfunction` only): flip
    the vertical axis; default `false` draws the surface at the top.
  - `coastlines`/`central_longitude` (`map_2d` only): per-component
    override of the shared `diagnostics.*` defaults above.
  - `var_opts.<variable>`: per-variable overrides — `cmap`, `vmin`, `vmax`,
    `symmetric`, `n_levels`, `units`.

`ocn.overturning_streamfunction` computes the meridional overturning
streamfunction (Sverdrups) from `v`/`dzt`/`mask_T`: zonally integrate
`v * dx * dz` per layer, using the standard MOC sign convention (surface
poleward-flow branches read positive). `ocn.heat_transport` computes the
meridional ocean heat transport (petawatts) from `v`/`temp`/`dzt`/`mask_T`:
`rho * cp * (integral of v*T over depth and longitude)`, using fixed
reference seawater constants (rho=1025 kg/m^3, cp=3990 J/(kg K)) since jem
does not report an equation of state. `atm.heat_transport` computes the
meridional atmospheric (sensible) heat transport (petawatts) from
`v_wind`/`temperature`/`normalized_surface_pressure`: `(c_pd/g) * (integral
of v*T over pressure and longitude)`, using SPEEDY's fixed sigma layer
thickness (7- or 8-level) to convert `normalized_surface_pressure` into a
per-layer pressure thickness. All three require a regular
(uniformly-spaced) longitude grid. Note `atm.heat_transport` uses
time-averaged `v`/`T` (mean(v)·mean(T)), so it only captures the
mean-circulation transport, not the transient-eddy covariance term
(mean(v'T')) that dominates real mid-latitude atmospheric heat transport —
expect it to look "peaky" near the equator (Hadley cell) and flat at
mid-to-high latitudes, unlike a textbook total-heat-transport curve.
- `output.base_dir`/`format`/`dpi`/`n_levels` — figure output defaults.
