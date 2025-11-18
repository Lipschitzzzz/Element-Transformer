import xarray as xr
import numpy as np

def nc2npz(nc_filename, npz_filename, parameters=['u', 'v']):
    ds = xr.open_dataset(nc_filename, decode_times=False)
    
    # ds['u'].values.shape (144, 40, 115443) (time, level, triangle)
    u = ds['u'].values
    v = ds['v'].values
    # print(u.shape)
    # print(u[:, 0, :].shape)
    # print(v[:, 0, :].shape)
    final_dataset = np.stack([u[:, 0, :], v[:, 0, :]], axis=-1)
    # print(final_dataset.shape)
    for i in range(0, final_dataset.shape[0]):
        print(npz_filename + 'step_' + str(i).zfill(3))
        print(final_dataset[i].shape)
        np.savez_compressed(npz_filename + 'step_' + str(i+144*2).zfill(3), data = final_dataset[i])


    ds.close()
    # np.savez_compressed(npz_filename, data = final_dataset)

if __name__ == "__main__":
    nc2npz('dataset/GGB_240508T00.nc', 'dataset/step_data/')