import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from load_data import load_data

# Physical constants
RHOCP = 1025.0 * 3994.0     # seawater density * specific heat [J m^-3 K^-1]
EARTH_RADIUS = 6.371e6       # [m]

# Dummy layer thickness [m] — replace with real dz once available in output
DUMMY_DZ = np.array([50.0, 70.0, 100.0, 140.0, 190.0, 240.0, 290.0, 340.0,
                     390.0, 440.0, 490.0, 540.0, 590.0, 640.0, 690.0])

parser = argparse.ArgumentParser()
parser.add_argument("--folder", type=str, required=True,
                    help="Path to simulation output folder (e.g. output_T31/long_run)")
parser.add_argument("--output", type=str, default="northward_transport.png",
                    help="Output figure filename")
parser.add_argument("--batch-range", type=int, nargs=2, metavar=("START", "END"),
                    default=None, help="Batch range to average over (inclusive). Default: all.")
parser.add_argument("--lon-range", type=float, nargs=2, metavar=("LON_START", "LON_END"),
                    default=None, help="Longitude range [degrees, 0-360]. Default: full circle.")
args = parser.parse_args()

batch_range = tuple(args.batch_range) if args.batch_range is not None else None

ds_atm = load_data(args.folder, "atm", batch_range)
ds_ocn = load_data(args.folder, "ocn", batch_range)

lon = ds_atm["lon"].values    # (nlon,)
lat = ds_atm["lat"].values    # (nlat,)
lat_rad = np.deg2rad(lat)

dlon_rad = np.deg2rad(lon[1] - lon[0])

# Longitude selection
if args.lon_range is not None:
    lon_start, lon_end = args.lon_range
    lon_mask = (lon >= lon_start) & (lon <= lon_end)
else:
    lon_mask = np.ones(len(lon), dtype=bool)
lon_indices = np.where(lon_mask)[0]
print(f"Integrating over {lon_indices.size} longitude points "
      f"({lon[lon_indices[0]]:.2f}° – {lon[lon_indices[-1]]:.2f}°)")

# 3D ocean mask at selected longitudes: (nlon_sel, nlat, ndepth)
mask_da = ds_ocn["mask_T"]
if "time" in mask_da.dims:
    mask_da = mask_da.isel(time=0)
mask = mask_da.values[lon_indices, :, :]    # (nlon_sel, nlat, ndepth)

# Time-mean fields at selected longitudes: (nlon_sel, nlat, ndepth)
v    = ds_ocn["v"].values.mean(axis=0)[lon_indices, :, :]
temp = ds_ocn["temp"].values.mean(axis=0)[lon_indices, :, :]
salt = ds_ocn["salt"].values.mean(axis=0)[lon_indices, :, :]

dz = DUMMY_DZ    # (ndepth,)

# Northward transport at each latitude:
#   OHT(lat) = RHOCP * R * cos(lat) * dλ * Σ_lon Σ_z  v * T * mask * dz
#   OST(lat) =          R * cos(lat) * dλ * Σ_lon Σ_z  v * S * mask * dz
#
# (v * field * mask * dz).sum(axis=(0,2)) → (nlat,), then multiply by R*cos(lat)*dλ

kernel = v * mask * dz[np.newaxis, np.newaxis, :]   # (nlon_sel, nlat, ndepth)

oht_W  = RHOCP * EARTH_RADIUS * dlon_rad * np.cos(lat_rad) * (kernel * temp).sum(axis=(0, 2))
ost    =         EARTH_RADIUS * dlon_rad * np.cos(lat_rad) * (kernel * salt).sum(axis=(0, 2))

oht_PW = oht_W * 1e-15
ost_Sv = ost   * 1e-6    # PSU·Sv

batch_label = (
    f"batches {args.batch_range[0]}–{args.batch_range[1]}"
    if args.batch_range is not None else "all batches"
)
lon_label = f"{lon[lon_indices[0]]:.1f}°–{lon[lon_indices[-1]]:.1f}°"
depth_label = f"dummy dz, total depth = {DUMMY_DZ.sum():.0f} m"

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

axes[0].plot(oht_PW, lat, color="darkorange")
axes[0].axvline(0, color="k", linewidth=0.8, linestyle="--")
axes[0].set_xlabel("OHT [PW]")
axes[0].set_ylabel("Latitude [°]")
axes[0].set_title(f"Northward ocean heat transport\n{lon_label}  ({batch_label})\n({depth_label})")
axes[0].grid(True, linestyle="--", alpha=0.5)

axes[1].plot(ost_Sv, lat, color="steelblue")
axes[1].axvline(0, color="k", linewidth=0.8, linestyle="--")
axes[1].set_xlabel("OST [PSU·Sv]")
axes[1].set_title(f"Northward ocean salt transport\n{lon_label}  ({batch_label})")
axes[1].grid(True, linestyle="--", alpha=0.5)

fig.tight_layout()
output_path = Path(args.output)
fig.savefig(output_path, dpi=150)
print(f"Saved: {output_path}")
