import xarray as xr
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib

def normalization(data):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    data_normalized = scaler.fit_transform(data)
    print("Original range:", data.min(axis=0), "->", data.max(axis=0))
    print("Normalized range:", data_normalized.min(axis=0), "->", data_normalized.max(axis=0))
    joblib.dump(scaler, '.scaler.pkl')
    # new_data_norm = joblib.load('scaler.pkl').transform(new_data)

def nc2npz(nc_filename, npz_filename, parameters=['u', 'v']):
    ds = xr.open_dataset(nc_filename, decode_times=False)
    
    # ds['u'].values.shape (144, 40, 115443) (time, level, triangle)
    u = ds['u'].values
    v = ds['v'].values
    scaler = MinMaxScaler(feature_range=(-1, 1))
    # print(u.shape)
    # print(u[:, 0, :].shape)
    # print(v[:, 0, :].shape)
    final_dataset = np.stack([u[:, 0, :], v[:, 0, :]], axis=-1)
    # print(final_dataset.shape)
    for i in range(0, final_dataset.shape[0]):
        print(npz_filename + 'step_' + str(i).zfill(3))
        print(final_dataset[i].shape)
        # normalization(final_dataset[i])
        data_normalized = scaler.fit_transform(final_dataset[i])
        # print("Original range:", data.min(axis=0), "->", data.max(axis=0))
        # print("Normalized range:", data_normalized.min(axis=0), "->", data_normalized.max(axis=0))
        joblib.dump(scaler,npz_filename + 'scaler/step_' + str(i+144*2).zfill(3) + '_scaler.pkl')
        np.savez_compressed(npz_filename + 'step_data/step_' + str(i+144*2).zfill(3), data = data_normalized)


    ds.close()
    # np.savez_compressed(npz_filename, data = final_dataset)

if __name__ == "__main__":
    nc2npz('dataset/GGB_240508T00.nc', 'dataset/')