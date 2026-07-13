# Developer notes

Notes for the next session (human or Claude) on how `jemdiag` got here and why
it's built the way it is. Not user-facing docs — see `README.md` for usage.

## History

- The original package (per-variable CLI: `plot_jem.py map|timeseries|zonal|
  profile --run-dir ... --component ... --variable ...`) was built on
  2026-06-12; see `old/DEVELOPER.md` for that history. Its `plot_map` never
  used cartopy at all, despite `CLAUDE.md` mandating cartopy for spatial
  maps — flagged there as an unresolved "convention mismatch."
- On 2026-07-13 the user moved that whole stack to `old/` and asked for a
  complete rewrite modeled on
  `/home/tienyiao/projects_local/plotting_poem`: a YAML config per
  experiment plus a single batch CLI (`plot_jem.py run --config ...`) that
  loops `component × diag_type` and dispatches through a per-component
  `HANDLERS` registry, backed by a shared cartopy-based `plot_utils.py`.
  This resolves the old cartopy gap as a side effect of adopting the
  pattern. `old/` is kept as read-only reference; nothing there is imported
  by the current code.
- `data/long_run` (this repo, not a symlink) has 9 files per component
  (`atm`, `ocn`, `fakelnd`), one degenerate `time` value per file
  (annual-snapshot style output, no usable time coordinate).

## Design decisions worth remembering

- **`ocn` now has real `lon`/`lat` coordinates** (`data/long_run/ocn-*.nc`:
  `lon`/`lat` are tagged `degrees_east`/`degrees_north`, not indices) —
  this differs from the assumption in `old/DEVELOPER.md` ("ocn has no real
  coordinates, by design, for now"), so a newer jem version apparently
  added them. Only `depth` still lacks coordinate values. Rather than
  hardcode a "no cartopy for ocn" rule, `jemdiag/diagnostics/ocn.py` checks
  `jemdiag.grid.has_real_horizontal_coords()` per variable at plot time and
  picks `plot_utils.global_map` (cartopy) or `plot_utils.index_map` (plain
  `pcolormesh`) accordingly — it will keep working correctly whether or not
  a future jem version changes this again, with zero config changes.
  Note: `ocn`'s `lat` array runs from -91.875 to 84.375 (half-cell overshoot
  past the poles, likely a tripolar-grid-to-1D-axis artifact) — harmless for
  plotting (cartopy just draws the row wherever `PlateCarree` maps it) but
  worth knowing about if it ever looks like a bug.
- **Ocean land cells are unmasked.** `ocn` global maps (e.g.
  `sea_surface_temperature`) show a couple of narrow, oddly-placed cold
  vertical stripes — enclosed-sea/land grid cells (e.g. Caspian/Black Sea
  longitudes) that carry a fill value rather than `NaN`, so `pcolormesh`
  draws them instead of leaving them transparent. `ocn`'s `mask_T`/
  `mask_surface_T` variables could mask these out; deliberately not wired in
  yet since neither the old code nor `plotting_poem`'s ocean diagnostics do
  this masking, and it wasn't asked for — flag to the user before adding it.
- **Time-axis normalization.** `jemdiag/data_loader.py:_normalize_time`
  checks the concatenated `time` index for uniqueness/monotonicity; if it's
  missing, duplicated, or non-monotonic (the `long_run` case), it's replaced
  with a synthetic record index `0..N-1` and tagged with a `note` attr. This
  now operates on the whole per-component `xr.Dataset` (loaded once via
  `xr.open_mfdataset`), not per-variable as in the old code, since a batch
  runner needs the full dataset available to iterate many variables/diag
  types without re-opening files each time.
- **`jemdiag_synthetic` attribute gates two things:** latitude-weighting in
  `compute.weighted_mean` (skipped for synthetic y-coords — uniform mean
  instead) and `selection.select_region` (raises `ValueError` if lon/lat are
  synthetic, since a bounding box in degrees is meaningless for index
  coords).
- **Variable titles/units come from the NetCDF attrs directly**
  (`description`/`long_name`/`units` on each `DataArray`), not a hand-
  maintained per-variable dict like `plotting_poem`'s `_UNITS` — jem's own
  files already carry clean metadata, so duplicating it in Python would just
  be another place for it to go stale. `ocn.py`'s `_var_label` strips a
  redundant leading "ocean " from `long_name` (e.g. "ocean potential
  temperature") before prefixing "Ocean " in titles.
- **No calendar / seasonal aggregation.** Unlike `plotting_poem`'s
  `temporal_agg.aggregate` (annual/DJF/MAM/JJA/SON/month), jem has no
  calendar time axis at all, so `time`/`level` in the config are `"mean"` |
  int | `"start:end"` slice over the raw record/level index
  (`jemdiag/selection.py`, carried over from the old CLI's semantics).

## Testing

```sh
cd jem_diag
python plot_jem.py list-vars --run-dir data/long_run --component atm
python plot_jem.py run --config config/long_run.yaml
```

The full `run --config config/long_run.yaml` pass (all 3 components, all 4
diag_types) was verified against `data/long_run`, and a sample of the
resulting PNGs was visually inspected: `atm`/`ocn` maps render on the correct
cartopy projection (no coastlines, per user request), `ocn`'s zonal/map
fallback logic was confirmed to pick cartopy now that its lon/lat are real,
and `fakelnd`'s curvilinear map, zonal sections, a vertical profile, and time
series all rendered correctly. `--components`/`--diag-types`/`--vars` filters
were spot-checked for
single-variable iteration.

## Pending / open items

- Ocean land-cell masking (see above) — not implemented, flag to user first.
- Only `PlateCarree` is implemented in `jemdiag/plot_utils.get_projection`-
  equivalent path (there isn't a projection factory yet since only one
  projection is needed today); add one if a polar or regional projection is
  ever needed.
