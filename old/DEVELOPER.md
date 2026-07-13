# Developer notes

Notes for the next session (human or Claude) on how `jemdiag` got here and why
it's built the way it is. Not user-facing docs — see `README.md` for usage.

## History

- The package (`jemdiag/`, `plot_jem.py`, `scripts/run_diagnostic.sh`,
  `requirements.txt`, `figures/`) was originally built on 2026-06-12 in
  `/home/tienyiao/projects_local/project_jax-esm/jem_repo/diagnostic-package-for-jem`,
  then relocated here the same day at the user's request ("I made a mistake in
  the location"). `diagnostic-package-for-jem` keeps its own git repo and a
  `jem_data` symlink but no longer contains any plotting code; see its own
  `DEVELOPER.md` for a pointer back here.
- `jem_data` here is a symlink to
  `../jem_repo/jax-esm/notebooks/02_experimental/02_experimental_JCM_Veros/output_T31/`,
  which contains two example run directories:
  - `test-highres/`: 3 files per component (`atm`, `ocn`, `fakelnd`), 365 daily
    steps/file, real `time` coordinate.
  - `long_run/`: 15 files per component, `time` has size 1 per file with no
    usable coordinate value (annual-snapshot style output).

## Design decisions worth remembering

- **ocn has no real coordinates (by design, for now).** `ocn`'s
  `longitude`/`latitude`/`depth` dims currently carry no coordinate values at
  all. Per explicit user instruction, these are treated as plain integer
  indices `0..N-1` (tagged `jemdiag_synthetic=True` in
  `jemdiag/data.py:_ensure_index_coords`) — do **not** borrow coordinates from
  the `atm` grid. Ocean output is expected to provide real coordinates itself
  in a future jem version; revisit this once it does.
- **Time-axis normalization.** `jemdiag/data.py:_normalize_time` checks the
  concatenated `time` index for uniqueness/monotonicity. If it's missing,
  duplicated, or non-monotonic (the `long_run` case — one degenerate value per
  file), it's replaced with a synthetic record index `0..N-1` and tagged with a
  `note` attr. `test-highres` has a real per-day datetime axis and passes
  through untouched. `plotting.plot_timeseries` adapts its x-axis label based
  on this `note` attr.
- **`jemdiag_synthetic` attribute gates two things:** latitude-weighting in
  `compute.weighted_mean` (skipped for synthetic y-coords — uniform mean
  instead) and `selection.select_region` (raises `ValueError` if lon/lat are
  synthetic, since a bounding box in degrees is meaningless for index coords).
- **fakelnd 2D curvilinear coords.** `fakelnd` has real `time` plus 2D
  `longitude2D`/`latitude2D` over `(longitude, latitude)`, matching the atm
  lon/lat grid. `grid.horizontal_coords` detects these by name + dims and
  returns them for `pcolormesh`, transposed to `(y_dim, x_dim)`.
- **Dotted variable names** (e.g. `shortwave_rad.rsds`, atm has 76 vars, many
  dotted) work fine as xarray data_var names; `plot_jem.py:_default_output`
  replaces `.` with `_` when building default output filenames.

## Convention mismatch — partially resolved

This repo's `CLAUDE.md` (written before this code existed) mandates
`xarray, numpy, matplotlib, cartopy`, with cartopy projections for maps.

- `jemdiag/compute.py` originally used `jax.numpy` (carried over from
  `diagnostic-package-for-jem`'s CLAUDE.md, which mandated `jax`). Switched to
  plain `numpy` on 2026-06-12 per user request ("I think numpy is better for
  plotting"); `requirements.txt` updated accordingly (`jax` -> `numpy`).
- `jemdiag/plotting.py` still uses plain matplotlib `pcolormesh`/`contourf`
  with no cartopy projections — still unresolved, raise with user if it
  matters for this project.

## Testing

```sh
cd jem_diag
./scripts/run_diagnostic.sh list-vars --run-dir jem_data/test-highres --component atm
./scripts/run_diagnostic.sh map --run-dir jem_data/long_run --component ocn \
    --variable sea_surface_temperature --output figures/smoketest.png
```

All four subcommands (`map`, `timeseries`, `zonal`, `profile`) plus `list-vars`
were verified against both `test-highres` and `long_run`, across all three
components, including edge cases: dotted variable names, index-based ocn
coords, fakelnd 2D coords, synthetic time indices, and error messages for
missing variables / invalid regions / missing vertical dims.

## Pending / open items

- Leftover files in `diagnostic-package-for-jem` (`.claude/`, `src/`,
  `CLAUDE.md`, `jem_data` symlink) — ask the user whether to delete that
  directory or leave it as-is.
- The jax/cartopy convention mismatch above is unresolved.
