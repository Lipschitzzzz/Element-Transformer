import xarray as xr
import numpy as np
import os
import json

def normalization(data, output_name, json_path):
    # data (144, 115443, 18)
    print("Original data shape:", data.shape)
    mins = data.min(axis=(0, 1))
    maxs = data.max(axis=(0, 1))
    eps = 1e-8
    ranges = maxs - mins + eps
    normalized_data = 2 * (data - mins) / ranges - 1
    print("Normalized min per channel:", normalized_data.min(axis=(0, 1)))
    print("Normalized max per channel:", normalized_data.max(axis=(0, 1)))
    np.save(output_name, normalized_data)
    print(f"Normalized data saved to {output_name}")
    norm_params = {
        "mins": mins.tolist(),
        "maxs": maxs.tolist(),
        "ranges": ranges.tolist(),
        "shape_before_normalization": list(data.shape),
        "normalized_range": [-1.0, 1.0]
    }
    with open(json_path, 'w') as f:
        json.dump(norm_params, f, indent=4)
    print(f"Normalization parameters saved to {json_path}")

def denormalization(normalized_data, json_path):
    with open(json_path, 'r') as f:
        params = json.load(f)
    mins = np.array(params['mins'])
    maxs = np.array(params['maxs'])
    ranges = maxs - mins
    if normalized_data.shape[-1] != 18:
        raise ValueError(f"Expected last dimension=18, got {normalized_data.shape[-1]}")
    reshape_shape = [1] * (normalized_data.ndim - 1) + [18]
    mins = mins.reshape(reshape_shape)
    ranges = ranges.reshape(reshape_shape)
    original_data = (normalized_data + 1.0) * ranges / 2.0 + mins
    return original_data

def nc2npy(nc_filename, parameters=['u', 'v']):
    ds = xr.open_dataset('dataset/' + nc_filename, decode_times=False)
    # ds['u'].values.shape (144, 40, 115443) (time, level, triangle)
    # for i in parameters:
    #     print(i)
    #     print(ds[i].shape)
    u = ds['u'].values[:, 0:40:8, :]
    v = ds['v'].values[:, 0:40:8, :]
    ww = ds['ww'].values[:, 0:40:8, :]

    tauc = ds['tauc'].values
    tauc = np.expand_dims(tauc, 1)
    uwind_speed = ds['uwind_speed'].values
    uwind_speed = np.expand_dims(uwind_speed, 1)
    vwind_speed = ds['vwind_speed'].values
    vwind_speed = np.expand_dims(vwind_speed, 1)
    print(u.shape, v.shape, ww.shape, tauc.shape, uwind_speed.shape, vwind_speed.shape)
    stacked = np.concatenate([u, v, ww, tauc, uwind_speed, vwind_speed], axis=1)
    print(stacked.shape)
    transposed = np.transpose(stacked, (0, 2, 1))
    print(transposed.shape)
    output_name = nc_filename.split('.')[0]
    normalization(transposed, 'dataset/step_data/' + output_name + '.npy',
                  'dataset/normalization_parameters/' + output_name + '.json')
    ds.close()

if __name__ == "__main__":
    vars = ['u', 'v', 'tauc', 'ww', 'uwind_speed', 'vwind_speed']
    file_list = os.listdir('dataset')
    for i in file_list:
        if i.endswith('.nc'):
            print(i)
            nc2npy(i, vars)