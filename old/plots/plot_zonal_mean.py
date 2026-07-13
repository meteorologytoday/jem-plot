import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from load_data import load_data

parser = argparse.ArgumentParser()
parser.add_argument("--folder", type=str, required=True,
                    help="Path to simulation output folder (e.g. output_T31/long_run)")
parser.add_argument("--output", type=str, default="zonal_mean.png",
                    help="Output figure filename")
parser.add_argument("--batch-range", type=int, nargs=2, metavar=("START", "END"),
                    default=None, help="Batch range to average over (inclusive). Default: all.")
parser.add_argument("--lon-range", type=float, nargs=2, metavar=("LON_START", "LON_END"),
                    default=None, help="Longitude range [degrees, 0-360] to average over. Default: full circle.")
args = parser.parse_args()

batch_range = tuple(args.batch_range) if args.batch_range is not None else None

ds_atm = load_data(args.folder, "atm", batch_range)
ds_ocn = load_data(args.folder, "ocn", batch_range)

lon = ds_atm["lon"].values    # (nlon,)
lat = ds_atm["lat"].values    # (nlat,)

# Longitude selection
if args.lon_range is not None:
    lon_start, lon_end = args.lon_range
    lon_mask = (lon >= lon_start) & (lon <= lon_end)
else:
    lon_mask = np.ones(len(lon), dtype=bool)
lon_indices = np.where(lon_mask)[0]
print(f"Averaging over {lon_indices.size} longitude points "
      f"({lon[lon_indices[0]]:.2f}° – {lon[lon_indices[-1]]:.2f}°)")

# Surface mask: (nlon, nlat) → select lon, replace 0 with NaN
mask_da = ds_ocn["mask_surface_T"]
if "time" in mask_da.dims:
    mask_da = mask_da.isel(time=0)
mask = mask_da.values[lon_indices, :].astype(float)   # (nlon_sel, nlat)
mask[mask == 0] = np.nan

# Time-mean then zonal masked mean
sst = ds_ocn["sea_surface_temperature"].values.mean(axis=0)[lon_indices, :] - 273.15  # (nlon_sel, nlat)
sss = ds_ocn["sea_surface_salinity"].values.mean(axis=0)[lon_indices, :]              # (nlon_sel, nlat)

# Masked zonal mean: mean over ocean points only
sst_zonal = np.nanmean(sst * mask, axis=0)   # (nlat,)
sss_zonal = np.nanmean(sss * mask, axis=0)   # (nlat,)

batch_label = (
    f"batches {args.batch_range[0]}–{args.batch_range[1]}"
    if args.batch_range is not None else "all batches"
)
lon_label = f"{lon[lon_indices[0]]:.1f}°–{lon[lon_indices[-1]]:.1f}°"

fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)

axes[0].plot(sst_zonal, lat, color="darkorange")
axes[0].axvline(0, color="k", linewidth=0.8, linestyle="--")
axes[0].set_xlabel("SST [°C]")
axes[0].set_ylabel("Latitude [°]")
axes[0].set_title(f"Zonal mean SST\n{lon_label}  ({batch_label})")
axes[0].grid(True, linestyle="--", alpha=0.5)

axes[1].plot(sss_zonal, lat, color="steelblue")
axes[1].set_xlabel("SSS [PSU]")
axes[1].set_title(f"Zonal mean SSS\n{lon_label}  ({batch_label})")
axes[1].grid(True, linestyle="--", alpha=0.5)

fig.tight_layout()
output_path = Path(args.output)
fig.savefig(output_path, dpi=150)
print(f"Saved: {output_path}")
