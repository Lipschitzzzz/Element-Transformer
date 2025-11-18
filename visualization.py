import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as ccrs

if __name__ == "__main__":
    ds = xr.open_dataset('dataset/GGB_240506T00.nc', decode_times=False)

    if 'lon' in ds and 'lat' in ds:
        lon = ds['lon'].values
        lat = ds['lat'].values
    else:
        raise KeyError("no 'lon'/'lat'")

    if 'nv' in ds:
        nv = ds['nv'].values
    else:
        raise KeyError("no 'nv'")

    triangles = (nv - 1).T if nv.min() == 1 else nv.T

    try:
        u = ds['u'].isel(time=0, siglay=-1).values
    except ValueError:
        u = ds['u'].isel(time=0).values

    assert len(u) == triangles.shape[0], f"u length {len(u)} and element {triangles.shape[0]} mismatch"

    lon_min, lon_max = lon.min(), lon.max()
    lat_min, lat_max = lat.min(), lat.max()

    delta_lon = lon_max - lon_min
    delta_lat = lat_max - lat_min
    width = 12
    height = max(4, min(12 * delta_lat / delta_lon, 10))

    plt.figure(figsize=(width, height))

    sc = plt.tripcolor(lon, lat, triangles, u, shading='flat')

    margin_lon = delta_lon * 0.02
    margin_lat = delta_lat * 0.02
    plt.xlim(lon_min - margin_lon, lon_max + margin_lon)
    plt.ylim(lat_min - margin_lat, lat_max + margin_lat)

    plt.xlabel('Longitude (°E)', fontsize=12)
    plt.ylabel('Latitude (°N)', fontsize=12)
    plt.title('FVCOM u-component on Triangular Mesh', fontsize=14)
    plt.colorbar(sc, shrink=0.8, label='u (m/s)')

    plt.tight_layout()
    plt.show()