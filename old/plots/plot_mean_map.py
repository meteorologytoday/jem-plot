import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cmocean as cmo
import cartopy.crs as ccrs

from load_data import load_data

parser = argparse.ArgumentParser()
parser.add_argument("--folder", type=str, required=True,
                    help="Path to simulation output folder (e.g. output_T31/long_run)")
parser.add_argument("--output", type=str, default="mean_map.png",
                    help="Output figure filename")
parser.add_argument("--batch-range", type=int, nargs=2, metavar=("START", "END"),
                    default=None, help="Batch range to average over (inclusive). Default: all.")
args = parser.parse_args()

batch_range = tuple(args.batch_range) if args.batch_range is not None else None

ds_atm = load_data(args.folder, "atm", batch_range)   # for lat/lon grid
ds_ocn = load_data(args.folder, "ocn", batch_range)

lon = ds_atm["lon"].values    # (nlon,)
lat = ds_atm["lat"].values    # (nlat,)

# Time-mean SST and SSS; mask land with NaN
mask_da = ds_ocn["mask_surface_T"]
if "time" in mask_da.dims:
    mask_da = mask_da.isel(time=0)
mask = mask_da.values.astype(float)   # (nlon, nlat), 1=ocean 0=land
mask[mask == 0] = np.nan

sst = ds_ocn["sea_surface_temperature"].values.mean(axis=0) - 273.15  # (nlon, nlat)
sss = ds_ocn["sea_surface_salinity"].values.mean(axis=0)      # (nlon, nlat)

sst_masked = sst * mask
sss_masked = sss * mask

# Near-surface (lowest model level) wind and specific humidity from the atmosphere
u_sfc = ds_atm["surface_flux.u0"].values.mean(axis=0)   # (nlon, nlat)
v_sfc = ds_atm["surface_flux.v0"].values.mean(axis=0)   # (nlon, nlat)
wind_speed = np.sqrt(u_sfc**2 + v_sfc**2)
q_sfc = ds_atm["specific_humidity"].isel(level=0).values.mean(axis=0)  # (nlon, nlat), g/kg

batch_label = (
    f"batches {args.batch_range[0]}–{args.batch_range[1]}"
    if args.batch_range is not None else "all batches"
)

fig, axes = plt.subplots(
    2, 2, figsize=(16, 10),
    subplot_kw=dict(projection=ccrs.PlateCarree()),
)

def plot_map(ax, data, lon, lat, levels, cmap, label, title, contour_levels=None,
             quiver_u=None, quiver_v=None, quiver_step=(4, 3), quiver_scale=200,
             quiver_key_speed=10):
    norm = mcolors.BoundaryNorm(levels, ncolors=cmap.N, extend="both")
    dlat = lat[1] - lat[0]
    dlon = lon[1] - lon[0]
    lat_bnd = np.concatenate([lat - 0.5*dlat, [lat[-1] + 0.5*dlat]])
    lon_bnd = np.concatenate([lon - 0.5*dlon, [lon[-1] + 0.5*dlon]])
    m = ax.pcolormesh(
        lon_bnd, lat_bnd, data.T,
        norm=norm, cmap=cmap, transform=ccrs.PlateCarree(),
    )
    plt.colorbar(m, ax=ax, orientation="vertical", shrink=0.7, pad=0.05, label=label)
    if contour_levels is not None:
        cs = ax.contour(
            lon, lat, data.T,
            levels=contour_levels, colors="k", linewidths=0.7,
            transform=ccrs.PlateCarree(),
        )
        ax.clabel(cs, fmt="%.0f", fontsize=7)
    if quiver_u is not None and quiver_v is not None:
        su, sv = quiver_step
        q = ax.quiver(
            lon[::su], lat[::sv],
            quiver_u[::su, ::sv].T, quiver_v[::su, ::sv].T,
            transform=ccrs.PlateCarree(), scale=quiver_scale, width=0.002, color="k",
        )
        ax.quiverkey(
            q, 0.9, -0.08, quiver_key_speed, f"{quiver_key_speed:g} m/s",
            labelpos="E", coordinates="axes", fontproperties=dict(size=7),
        )
    ax.gridlines(draw_labels=True, linewidth=0.5, linestyle="--", alpha=0.5)
    ax.set_title(title)

plot_map(
    axes[0, 0], sst_masked, lon, lat,
    levels=np.linspace(-2, 32, 35),
    cmap=cmo.cm.thermal,
    label="deg C",
    title=f"Mean SST ({batch_label})",
    contour_levels=np.arange(0, 32, 4),
)

plot_map(
    axes[0, 1], sss_masked, lon, lat,
    levels=np.linspace(33, 38, 26),
    cmap=cmo.cm.haline,
    label="PSU",
    contour_levels=np.arange(33, 38, 0.5),
    title=f"Mean SSS ({batch_label})",
)

plot_map(
    axes[1, 0], wind_speed, lon, lat,
    levels=np.linspace(0, 15, 16),
    cmap=cmo.cm.speed,
    label="m/s",
    title=f"Mean Surface Wind ({batch_label})",
    quiver_u=u_sfc, quiver_v=v_sfc,
)

plot_map(
    axes[1, 1], q_sfc, lon, lat,
    levels=np.linspace(0, 20, 21),
    cmap=cmo.cm.rain,
    label="g/kg",
    title=f"Mean Surface Specific Humidity ({batch_label})",
)

fig.tight_layout()
output_path = Path(args.output)
fig.savefig(output_path, dpi=150)
print(f"Saved: {output_path}")
