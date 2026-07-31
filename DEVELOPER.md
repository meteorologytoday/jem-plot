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
- **`ocn`'s `depth` index runs seafloor → surface, not surface → seafloor.**
  Confirmed on 2026-07-26 by comparing `temp`'s per-depth-index mean against
  `sea_surface_temperature`: they match at the *last* depth index, and
  `dzt` (layer thickness) decreases monotonically toward that same end
  (690m at index 0 down to 50m at the last index) — thin surface layers,
  thick abyssal layers, index 0 = seafloor. This had been backwards in
  `config/long_run.yaml` (`ocn.zonal_section.invert_yaxis: true`), which
  drew `temp`/`salt` sections with the warm surface at the *bottom* of the
  chart. Fixed by defaulting `invert_yaxis` to `false` in
  `jemdiag/diagnostics/ocn.py:_run_zonal_section` and in the config, so
  plain (uninverted) matplotlib axis order — index 0 at the bottom, last
  index at top — now correctly puts the surface on top.
- **`overturning_streamfunction`/`heat_transport` (`jemdiag/compute.py`,**
  **added 2026-07-26).** Two new `ocn` diag_types, derived quantities with
  no `vars` list (unlike the per-variable diag_types) since each is a fixed
  combination of raw fields:
  - `overturning_streamfunction`: psi(depth, lat) in Sv — zonally integrate
    `v * dx * dz` per layer (`dx = R*cos(lat)*dlon`, requires a regular
    longitude grid — `compute._uniform_spacing` raises if not), then
    cumulative-sum from the seafloor (depth index 0) upward. This is the
    naive combined barotropic+overturning streamfunction (no gyre/
    barotropic component removed), so magnitudes (~100+ Sv in
    `data/long_run`) run well above a decomposed AMOC-only number
    (~10-20 Sv) — flagged to the user, not a bug.
  - `heat_transport`: MHT(lat) in PW — `rho*cp * integral of v*T over depth
    and longitude`, using fixed reference seawater constants (rho=1025
    kg/m^3, cp=3990 J/(kg K)) since jem reports no equation of state; `temp`
    (potential temperature, degC) is used directly rather than converting
    to Kelvin, which is standard practice for this diagnostic since the net
    volume transport across a latitude circle is ~0. Sanity-checked against
    `data/long_run`: peaks around ±1-3 PW with the expected equator-to-pole
    sign pattern, the right order of magnitude for a real ocean.
  - Both multiply by `mask_T` before the zonal sum, even though `v`/`temp`
    already happen to be exactly `0` on land in `data/long_run` — don't rely
    on that implicitly, a future jem run's land fill value isn't guaranteed
    to be `0`. This is a different use of `mask_T` than the deliberately-
    unwired map-masking case above (that's about hiding land visually in a
    horizontal map; this is about correctness of a physical integral), so it
    doesn't contradict that pending item.
- **No calendar / seasonal aggregation.** Unlike `plotting_poem`'s
  `temporal_agg.aggregate` (annual/DJF/MAM/JJA/SON/month), jem has no
  calendar time axis at all, so `time`/`level` in the config are `"mean"` |
  int | `"start:end"` slice over the raw record/level index
  (`jemdiag/selection.py`, carried over from the old CLI's semantics).
- **Multi-snapshot `time` + shared default.** For every non-timeseries
  diag_type, `time` may also be a list of specs (`selection.
  normalize_time_args`), producing one output file per entry with a
  `_<time_label>` filename suffix (`selection.time_label`, e.g. `t0`,
  `t5-10`, `mean`) instead of the plain `{variable}.{fmt}` name used for a
  single spec. `diagnostics.time` is a shared default resolved by
  `selection.resolve_time_spec`; each diag_type block can still set its own
  `time` to override it. `level` was deliberately left single-spec-only —
  nothing asked for a level sweep, and multiplying it against a list of
  `time` specs would blow up combinatorially for no requested benefit.

## Testing

```sh
cd jem_diag
python plot_jem.py list-vars --run-dir data/long_run --component atm
python plot_jem.py run --config config/long_run.yaml
```

The full `run --config config/long_run.yaml` pass (all 3 components, all 4
original diag_types) was verified against `data/long_run`, and a sample of the
resulting PNGs was visually inspected: `atm`/`ocn` maps render on the correct
cartopy projection (no coastlines, per user request), `ocn`'s zonal/map
fallback logic was confirmed to pick cartopy now that its lon/lat are real,
and `fakelnd`'s curvilinear map, zonal sections, a vertical profile, and time
series all rendered correctly. `--components`/`--diag-types`/`--vars` filters
were spot-checked for single-variable iteration.

On 2026-07-26, `ocn.overturning_streamfunction` and `ocn.heat_transport`
were added and re-verified end-to-end (`--components ocn --diag-types
zonal_section overturning_streamfunction heat_transport`); output PNGs were
visually inspected (see above for the sanity-check results) alongside the
`invert_yaxis` orientation fix for `zonal_section`.

## Pending / open items

- Ocean land-cell masking (see above) — not implemented, flag to user first.
- Only `PlateCarree` is implemented in `jemdiag/plot_utils.get_projection`-
  equivalent path (there isn't a projection factory yet since only one
  projection is needed today); add one if a polar or regional projection is
  ever needed.
