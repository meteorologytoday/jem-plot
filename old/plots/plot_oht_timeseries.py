import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from load_data import load_data

# Physical constants
RHOCP = 1025.0 * 3994.0     # seawater density * specific heat [J m^-3 K^-1]
EARTH_RADIUS = 6.371e6       # [m]

# Dummy layer thickness [m] — replace with real dz once it is available in output
DUMMY_DZ = np.array([50.0, 70.0, 100.0, 140.0, 190.0, 240.0, 290.0, 340.0, 390.0, 440.0, 490.0, 540.0, 590.0, 640.0, 690.0])

parser = argparse.ArgumentParser()
parser.add_argument("--folder", type=str, required=True,
                    help="Path to simulation output folder (e.g. output_T31/long_run)")
parser.add_argument("--output", type=str, default="oht_timeseries.png",
                    help="Output figure filename")
parser.add_argument("--batch-range", type=int, nargs=2, metavar=("START", "END"),
                    default=None, help="Batch range to load (inclusive)")
parser.add_argument("--latitude", type=float, required=True,
                    help="Target latitude [degrees] at which to evaluate OHT")
parser.add_argument("--lon-range", type=float, nargs=2, metavar=("LON_START", "LON_END"),
                    default=None, help="Longitude range [degrees, 0-360] to integrate over. Default: full circle.")
args = parser.parse_args()

batch_range = tuple(args.batch_range) if args.batch_range is not None else None

ds_atm = load_data(args.folder, "atm", batch_range)   # for lat/lon grid
ds_ocn = load_data(args.folder, "ocn", batch_range)

# Grid from ATM (shared horizontal grid with OCN)
lon = ds_atm["lon"].values    # (nlon,) degrees, 0–360
lat = ds_atm["lat"].values    # (nlat,) degrees

dlon_rad = np.deg2rad(lon[1] - lon[0])   # uniform zonal spacing [rad]

# Find target latitude index
lat_idx = int(np.argmin(np.abs(lat - args.latitude)))
lat_used = lat[lat_idx]
print(f"Requested latitude: {args.latitude:.2f}°, using nearest: {lat_used:.4f}°")

# Longitude index range
if args.lon_range is not None:
    lon_start, lon_end = args.lon_range
    lon_mask = (lon >= lon_start) & (lon <= lon_end)
else:
    lon_mask = np.ones(len(lon), dtype=bool)

lon_indices = np.where(lon_mask)[0]
print(f"Integrating over {lon_indices.size} longitude points "
      f"({lon[lon_indices[0]]:.2f}° – {lon[lon_indices[-1]]:.2f}°)")

# OCN mask at target latitude: (nlon, ndepth)
mask_da = ds_ocn["mask_T"]
if "time" in mask_da.dims:
    mask_da = mask_da.isel(time=0)
mask_lat = mask_da.values[lon_indices, lat_idx, :]    # (nlon_sel, ndepth)

# v, temp, salt at target latitude: (time, nlon_sel, ndepth)
v    = ds_ocn["v"].values[:,    lon_indices, lat_idx, :]
temp = ds_ocn["temp"].values[:, lon_indices, lat_idx, :]
salt = ds_ocn["salt"].values[:, lon_indices, lat_idx, :]

# dx for each selected longitude point [m]
lat_rad = np.deg2rad(lat_used)
dx = EARTH_RADIUS * np.cos(lat_rad) * dlon_rad    # scalar (uniform grid)

dz = DUMMY_DZ    # (ndepth,)

masked   = mask_lat[np.newaxis, :, :]                          # (1, nlon_sel, ndepth)
dz_bcast = dz[np.newaxis, np.newaxis, :]                       # (1, 1, ndepth)

# OHT [W] = RHOCP * dx * Σ_lon Σ_z  v * T * mask * dz
oht_W  = RHOCP * dx * (v * temp * masked * dz_bcast).sum(axis=(1, 2))
oht_PW = oht_W * 1e-15

# OST [PSU·m³/s] = dx * Σ_lon Σ_z  v * S * mask * dz  (÷1e6 → PSU·Sv)
ost    = dx * (v * salt * masked * dz_bcast).sum(axis=(1, 2))
ost_Sv = ost * 1e-6

time = ds_ocn["time"].values
lon_label = f"lon {lon[lon_indices[0]]:.1f}°–{lon[lon_indices[-1]]:.1f}°"
depth_label = f"dummy dz, total depth = {DUMMY_DZ.sum():.0f} m"

fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

axes[0].plot(time, oht_PW, color="darkorange")
axes[0].axhline(0, color="k", linewidth=0.8, linestyle="--")
axes[0].set_ylabel("OHT [PW]")
axes[0].set_title(f"Meridional ocean heat transport at {lat_used:.2f}°  —  {lon_label}  ({depth_label})")
axes[0].grid(True, linestyle="--", alpha=0.5)

axes[1].plot(time, ost_Sv, color="steelblue")
axes[1].axhline(0, color="k", linewidth=0.8, linestyle="--")
axes[1].set_ylabel("OST [PSU·Sv]")
axes[1].set_title(f"Meridional ocean salt transport at {lat_used:.2f}°  —  {lon_label}")
axes[1].grid(True, linestyle="--", alpha=0.5)

fig.tight_layout()

output_path = Path(args.output)
fig.savefig(output_path, dpi=150)
print(f"Saved: {output_path}")
