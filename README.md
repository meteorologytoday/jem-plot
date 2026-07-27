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
  `PlateCarree` projection (no coastlines drawn).
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
`figures/long_run/atm/map_2d/shortwave_rad_rsds.png`.

## Config file schema

See `config/long_run.yaml` for a complete example. Top-level sections:

- `experiment.name` — used in output paths.
- `data.run_dir` — directory containing `{component}-*.nc` files.
- `grids.<component>.projection` — cartopy projection name (currently only
  `PlateCarree` is implemented); horizontal/vertical dimension names and
  real-vs-synthetic coordinates are auto-detected per variable, so `ocn`
  needs no grid entry at all.
- `diagnostics.<component>.<diag_type>` — one block per
  `map_2d`/`zonal_section`/`profile`/`timeseries`/`overturning_streamfunction`/
  `heat_transport` (components only produce the diag_types listed in their
  `HANDLERS`, see `jemdiag/diagnostics/`):
  - `vars`: list of variable names to plot. Not used by
    `overturning_streamfunction`/`heat_transport`, which are single derived
    quantities computed from fixed inputs (`v`/`dzt`/`mask_T`, and
    `v`/`temp`/`dzt`/`mask_T` respectively) rather than a user-selected
    variable.
  - `time`/`level`: `"mean"`, an integer index, or a `"start:end"` slice
    (averaged over the slice). jem runs have no calendar, so this replaces
    season/month aggregation — it always operates on raw record index.
  - `region` (`profile`/`timeseries` only): `[lon_min, lon_max, lat_min,
    lat_max]` bounding box; requires real lon/lat coordinates.
  - `invert_yaxis` (`zonal_section`/`overturning_streamfunction` only): flip
    the vertical axis; default `false` draws the surface at the top.
  - `var_opts.<variable>`: per-variable overrides — `cmap`, `vmin`, `vmax`,
    `symmetric`, `n_levels`, `units`.

`ocn.overturning_streamfunction` computes the meridional overturning
streamfunction (Sverdrups) from `v`/`dzt`/`mask_T`: zonally integrate
`v * dx * dz` per layer, then cumulatively sum from the seafloor upward.
`ocn.heat_transport` computes the meridional ocean heat transport
(petawatts) from `v`/`temp`/`dzt`/`mask_T`: `rho * cp * (integral of v*T
over depth and longitude)`, using fixed reference seawater constants
(rho=1025 kg/m^3, cp=3990 J/(kg K)) since jem does not report an equation
of state. Both require a regular (uniformly-spaced) longitude grid.
- `output.base_dir`/`format`/`dpi`/`n_levels` — figure output defaults.
