import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from load_data import load_data

ATM_SURFACE_TEMP_VAR = "surface_flux.tsfc"
OCN_SST_VAR = "sea_surface_temperature"

parser = argparse.ArgumentParser()
parser.add_argument("--folder", type=str, required=True, help="Path to simulation output folder (e.g. output_T31/long_run)")
parser.add_argument("--output", type=str, default="timeseries.png", help="Output figure filename")
parser.add_argument("--batch-range", type=int, nargs=2, metavar=("START", "END"), default=None, help="Batch range to load (inclusive)")
args = parser.parse_args()

batch_range = tuple(args.batch_range) if args.batch_range is not None else None

ds_atm = load_data(args.folder, "atm", batch_range)
ds_ocn = load_data(args.folder, "ocn", batch_range)

# ATM: dims (time, lon, lat); OCN: dims (time, longitude, latitude) with no lat coord.
# Use ATM's lat for weighting both (same grid).
lat = ds_atm["lat"].values                          # (nlat,)
weights = np.cos(np.deg2rad(lat))
weights /= weights.mean()

t_surf = ds_atm[ATM_SURFACE_TEMP_VAR].values        # (time, lon, nlat)
sst    = ds_ocn[OCN_SST_VAR].values                 # (time, longitude, nlat)
mask_da = ds_ocn["mask_surface_T"]
if "time" in mask_da.dims:
    mask_da = mask_da.isel(time=0)
mask = mask_da.values                               # (longitude, nlat), 1=ocean 0=land

weights_atm = weights[np.newaxis, :]                          # (1, nlat)
weights_ocn = weights[np.newaxis, :] * mask                   # (longitude, nlat)

global_t_surf = (t_surf * weights_atm[np.newaxis]).mean(axis=(1, 2)) - 273.15
global_sst    = (sst * weights_ocn[np.newaxis]).sum(axis=(1, 2)) / weights_ocn.sum() - 273.15

time_atm = ds_atm["time"].values
time_ocn = ds_ocn["time"].values

fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

axes[0].plot(time_ocn, global_sst, color="steelblue")
axes[0].set_ylabel("SST [°C]")
axes[0].set_title("Global mean sea surface temperature")
axes[0].grid(True, linestyle="--", alpha=0.5)

axes[1].plot(time_atm, global_t_surf, color="firebrick")
axes[1].set_ylabel("T [°C]")
axes[1].set_title("Global mean surface air temperature")
axes[1].grid(True, linestyle="--", alpha=0.5)

fig.tight_layout()
output_path = Path(args.output)
fig.savefig(output_path, dpi=150)
print(f"Saved: {output_path}")
