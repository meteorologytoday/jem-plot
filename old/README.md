# diagnostic-package-for-jem

Flexible diagnostic plotting for output produced by `jax-esm` (jem).

## Layout

```
jemdiag/            # reusable plotting library
├── data.py         # discover/load run output, concatenate across files
├── grid.py          # dimension/coordinate detection (atm/ocn/fakelnd)
├── compute.py        # area-weighted reductions (numpy)
├── selection.py       # time/level/region selection & reduction
└── plotting.py        # matplotlib plot functions (map, timeseries, zonal, profile)
plot_jem.py        # argparse CLI entry point
scripts/
└── run_diagnostic.sh  # bash wrapper around plot_jem.py
requirements.txt   # numpy, xarray, netCDF4, matplotlib
```

A run directory contains numbered NetCDF files per component, e.g.
`atm-00000.nc`, `ocn-00000.nc`, `fakelnd-00000.nc`, .... `plot_jem.py`
concatenates these along `time` automatically.

Notes on grids:
- `atm` has real `lon`/`lat`/`level` coordinates.
- `ocn` currently has no coordinate values; `longitude`/`latitude`/`depth`
  are treated as plain integer indices (real ocean coordinates will be
  added in a future jem version).
- `fakelnd` has real `time` plus 2D curvilinear `longitude2D`/`latitude2D`.
- If a run's `time` coordinate is missing or degenerate (e.g. one
  non-informative value per file), it is replaced with a synthetic record
  index (0, 1, 2, ...).

## Usage

List available variables (atm has many, with dotted names like
`shortwave_rad.rsds`):

```sh
scripts/run_diagnostic.sh list-vars --run-dir jem_data/test-highres --component atm
```

Horizontal map (time- and level-mean by default; figures saved under
`figures/` unless `--output` is given):

```sh
scripts/run_diagnostic.sh map \
    --run-dir jem_data/test-highres --component atm \
    --variable temperature --time mean --level mean
```

Domain-mean time series:

```sh
scripts/run_diagnostic.sh timeseries \
    --run-dir jem_data/test-highres --component ocn \
    --variable sea_surface_temperature
```

Zonal-mean cross section (latitude vs. level/depth):

```sh
scripts/run_diagnostic.sh zonal \
    --run-dir jem_data/test-highres --component atm \
    --variable temperature --time mean
```

Domain-mean vertical profile:

```sh
scripts/run_diagnostic.sh profile \
    --run-dir jem_data/test-highres --component atm \
    --variable specific_humidity --time mean
```

`--time`/`--level` accept `mean`, an integer index, or a `start:end` slice
(averaged over the slice). `--region LON_MIN LON_MAX LAT_MIN LAT_MAX`
restricts `timeseries`/`profile` to a bounding box (requires a component
with real lon/lat coordinates, i.e. `atm` or `fakelnd`).
