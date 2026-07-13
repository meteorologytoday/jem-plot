import xarray as xr
from pathlib import Path


def load_data(folder, component, batch_range=None):
    files = sorted(Path(folder).glob(f"{component}-*.nc"))

    if batch_range is not None:
        start, end = batch_range
        files = [f for f in files if start <= int(f.stem.split("-")[1]) <= end]

    if not files:
        raise FileNotFoundError(f"No files found for component '{component}' in {folder}")

    ds = xr.open_mfdataset(files, concat_dim="time", combine="nested")
    return ds

