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

On 2026-07-31, several fixes/additions were made and re-verified against
`data/long_run`:
- **Land-fill contrast bug.** `temp`/`salt` (and any ocn field sharing
  `mask_T`'s full grid) hold `0` on land rather than NaN, which sat inside
  the real data range and washed out contrast (worst for `salt`, whose
  land-`0` is far below any real salinity) — fixed via
  `selection.mask_land`, applied in `ocn.py`'s `map_2d`/`zonal_section`/
  `profile`/`timeseries` handlers.
- **Physical ocean depth.** `zonal_section`/`profile`/
  `overturning_streamfunction` now plot real depth (m, negative below the
  surface, via `grid.ocean_depth_from_dzt`) instead of the bare layer
  index, falling back to the index if `dzt` is missing.
- **Lat/lon gridlines** added to `global_map` (cartopy `gridlines(draw_labels=True)`) — previously map_2d plots had no coordinate labels at all.
- **Overturning streamfunction sign fix.** Was integrating bottom-up with
  no sign flip, giving the opposite sign from the standard MOC convention
  (a poleward-surface/equatorward-deep cell should read positive) — see
  `compute.overturning_streamfunction`'s docstring for the derivation.
  Cross-checked against `ocn.heat_transport` (which has no such ambiguity,
  being a direct integral, not a cumsum): the fixed sign now agrees with
  heat transport's positive-in-NH result for a poleward-surface-flow cell.
- **`atm.heat_transport`** added: meridional atmospheric (sensible) heat
  transport, mirroring `ocn.heat_transport`'s structure but integrating
  over SPEEDY's sigma coordinate via `normalized_surface_pressure` and the
  fixed 7-/8-level sigma half-level table (`compute.
  SPEEDY_SIGMA_HALF_LEVELS`, matching `jcm.physics.speedy.
  physical_constants.SIGMA_LAYER_BOUNDARIES` — full-level sigma spacing
  does not recover these boundaries, so they're reproduced exactly rather
  than approximated). Uses time-averaged `v`/`T`, so it captures only the
  mean-circulation transport, not the transient-eddy term that dominates
  real mid-latitude atmospheric heat transport — expect a curve that's
  peaky near the equator (Hadley cell) and flat at mid/high latitudes.
- **`lon_bands`** added to `ocn.overturning_streamfunction`/
  `ocn.heat_transport`, restricting the zonal integral to a longitude
  sub-range (e.g. one ocean basin); accepts a single `[lon_min, lon_max]`
  or a list of them for one plot per band, same list-expansion pattern as
  `time` (`selection.normalize_lon_bands`/`select_lon_band`).
- **`sea_surface_salinity`** added to `ocn.map_2d.vars` in
  `config/long_run.yaml` (already had a `DEFAULT_CMAPS` entry, just wasn't
  wired into the example config).
- **`apply_var_filter` bug fix** (`plot_jem.py`): iterated over every
  value in `config["diagnostics"]` assuming all were per-component dicts;
  broke (`AttributeError`) once `diagnostics.time` (a shared scalar
  default, added earlier the same day) was introduced. Now skips
  non-dict entries.

On 2026-09-02, a **`coastlines` toggle** was added for `map_2d`
(`plot_utils.resolve_coastlines` -> `plot_utils.global_map`, wired into
`atm`/`ocn`/`fakelnd`'s `_run_map_2d`). Resolution order mirrors `time`:
diag_type block -> shared `diagnostics.coastlines` -> off. Notes:
- **Off by default**, reversing the earlier "no coastlines" decision only
  where a config opts in: jem runs are idealized unless configured with
  real Earth topography, and drawing Earth's coastline over an
  aqua-planet/idealized land-sea mask would be actively misleading. Enabled
  in `config/real_topo-reglatlon.yaml` only.
- The config value is a bool, or a mapping restricted to exactly
  `resolution`/`color`/`linewidth` (`plot_utils.COASTLINE_DEFAULTS`); any
  other key, a bad `resolution`, or a non-mapping/non-bool raises
  `ValueError` rather than silently drawing nothing. `zorder=2.5` is set by
  `global_map` itself (not configurable) so the outline always sits above
  the `pcolormesh`.
- Coastlines are a cartopy-only feature, so `ocn.map_2d` prints an
  "ignoring `coastlines`" note when a variable falls back to `index_map`
  for lack of real coordinates, instead of failing.
- Natural Earth shapefiles are fetched by cartopy on first use and cached
  under `~/.local/share/cartopy`; all three resolutions were already
  present on this machine, so enabling the toggle needs no network here.

The same day, a **`central_longitude` option** was added for `map_2d`
(`plot_utils.resolve_central_longitude`), sharing the resolution order above
via `plot_utils._resolve_map_option`. It's passed to the existing
`get_projection` factory as a `PlateCarree` kwarg, so only the projection is
recentered — the data is untouched and cartopy wraps the cells straddling
the new map edge. Set to `200` in `config/real_topo-reglatlon.yaml` (keeps
the Pacific whole). Like `coastlines` it's cartopy-only, so `ocn.map_2d`
lists it in the same "ignoring ..." note when a variable falls back to
`index_map`.

## Pending / open items

- Ocean land-cell masking (see above) — not implemented, flag to user first.
- Only `PlateCarree` is implemented in `jemdiag/plot_utils.get_projection`-
  equivalent path (there isn't a projection factory yet since only one
  projection is needed today); add one if a polar or regional projection is
  ever needed.
