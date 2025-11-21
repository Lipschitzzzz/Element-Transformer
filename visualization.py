import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as ccrs
import elementtransformer
import torch
import processing
import json

def count_parameters(model):
    for p in model.parameters():
        print(f"p: {p.numel()}")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    return total_params, trainable_params

def two_img(title, lon, lat, lon_min, lon_max, lat_min, lat_max,
            delta_lon, delta_lat, triangles, target_data, output,
            min_value=99, max_value=-99, time='1'):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

    sc1 = ax1.tripcolor(lon, lat, triangles, target_data, vmin=min_value, vmax=max_value, shading='flat')

    margin_lon = delta_lon * 0.02
    margin_lat = delta_lat * 0.02
    ax1.set_xlim(lon_min - margin_lon, lon_max + margin_lon)
    ax1.set_ylim(lat_min - margin_lat, lat_max + margin_lat)

    ax1.set_xlabel('Longitude (°E)', fontsize=12)
    ax1.set_ylabel('Latitude (°N)', fontsize=12)
    ax1.set_title('u-component layer 1 FVCOM 2024 05 06 ' + time, fontsize=14)
    fig.colorbar(sc1, ax=ax1, shrink=0.8, label='(m/s)')

    sc2 = ax2.tripcolor(lon, lat, triangles, output, vmin=min_value, vmax=max_value, shading='flat')

    ax2.set_xlim(lon_min - margin_lon, lon_max + margin_lon)
    ax2.set_ylim(lat_min - margin_lat, lat_max + margin_lat)

    ax2.set_xlabel('Longitude (°E)', fontsize=12)
    ax2.set_ylabel('Latitude (°N)', fontsize=12)
    ax2.set_title('u-component layer 1 AI 2024 05 06 ' + time, fontsize=14)
    fig.colorbar(sc2, ax=ax2, shrink=0.8, label='(m/s)')

    plt.savefig(title + '.jpg', dpi=300, bbox_inches='tight')
    plt.tight_layout()
    plt.close()
    # plt.show()

def int_to_time_str(i: int) -> str:
    if not (0 <= i <= 143):
        raise ValueError("Input i must be between 0 and 142")
    total_minutes = i * 10 + 10
    hours = total_minutes // 60
    minutes = total_minutes % 60
    return f"{hours:02d}:{minutes:02d}"
    
def predict_by_step(model, checkpoint_name, input_data):
    output = model.predict(input_data=input_data, checkpoint_name=checkpoint_name)
    # [B, T, E, C]
    return output

def predict_one_day(npy_file, model, checkpoint_name, device):
    checkpoint_name=checkpoint_name
    all_data = np.load(npy_file)
    criterion = elementtransformer.WeightedMAEMSELoss().cuda()
    all_output = []
    daily_length = 3
    predict_length = 1
    for i in range(0, daily_length - predict_length):
        step = i
        input_data = torch.tensor(all_data[step,:,:])
        target_data = torch.from_numpy(all_data[step+predict_length,:,:]).unsqueeze(0).unsqueeze(0).to(device)
        input_data = input_data.unsqueeze(0).unsqueeze(0).to(device)
        output = model.predict(input_data=input_data, checkpoint_name=checkpoint_name)
        print('loss: ', criterion(output, target_data))
        # print(output.shape)
        all_output.append(output.squeeze(0).squeeze(0).cpu().numpy())
    output = processing.denormalization(np.array(all_output), "dataset/normalization_parameters/GGB_240506T00.json")
    # (144, 115443, 18)
    return output
def init_config(nc_file, json_path):
    ds = xr.open_dataset(nc_file, decode_times=False)
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
    # config = [lon, lat, lon_min, lon_max, lat_min, lat_max,
    #             delta_lon, delta_lat, triangles]
    config = {
        "lon": lon.tolist(),
        "lat": lat.tolist(),
        "lon_min": lon_min.tolist(),
        "lon_max": lon_max.tolist(),
        "lat_min": lat_min.tolist(),
        "lat_max": lat_max.tolist(),
        "delta_lon": delta_lon.tolist(),
        "delta_lat": delta_lat.tolist(),
        "triangles": triangles.tolist()
    }
    with open(json_path, 'w') as f:
        json.dump(config, f, indent=4)
    

def img_drawing_by_batch(nc_file, layer, pred, target):
    daily_length = 3
    predict_length = 1
    # print(pred.shape, target.shape)
    # pred target 144, 115443, 18
    min_value = min(np.nanmin(pred[:,:,layer]), np.nanmin(target[:,:,layer]))
    max_value = max(np.nanmax(pred[:,:,layer]), np.nanmax(target[:,:,layer]))
    print('min_value:', min_value)
    print('max_value:', max_value)
    with open('img_config.json', 'r') as f:
        params = json.load(f)
    lon = np.array(params['lon'])
    lat = np.array(params['lat'])
    lon_min = params['lon_min']
    lon_max = params['lon_max']
    lat_min = params['lat_min']
    lat_max = params['lat_max']
    delta_lon = params['delta_lon']
    delta_lat = params['delta_lat']
    triangles = np.array(params['triangles'])
    for step in range(0, daily_length - predict_length):
        two_img('result/' + str(step).zfill(3), lon, lat, lon_min, lon_max, lat_min, lat_max,
            delta_lon, delta_lat, triangles, target[step,:,layer], pred[step,:,layer], min_value, max_value, int_to_time_str(step))
    

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    nc_file = 'dataset/GGB_240511T00.nc'
    checkpoint_name='checkpoints/2025_11_21_11_27_Inha_A100_GPU_Server_model.pth'
    npy_file='dataset/step_data/GGB_240511T00.npy'
    model = elementtransformer.FVCOMModel(var_in=18,var_out=18,triangle=115443, embed_dim=256, t_in=1).to(device)
    one_day_prediction_data = predict_one_day(npy_file, model, checkpoint_name, device)
    all_data = np.load(npy_file)[1:]
    target_data = processing.denormalization(all_data, "dataset/normalization_parameters/GGB_240506T00.json")
    img_drawing_by_batch(nc_file, 0, one_day_prediction_data, target_data[:4,:,:])
    # print(min_value)
    # print(max_value)
    # init_config(nc_file, 'img_config.json')

    # for i in range(0, 144):
    #     # pred_list.append(output)
    #     # target_list.append(target_data)
    #     print(output.shape)
    #     print(target_data.shape)
    #     print('step: ', step)

    #     print('min_value output: ', np.nanmin(output[:,0]))
    #     print('max_value output: ', np.nanmax(output[:,0]))

    #     print('min_value target_data: ', np.nanmin(target_data[:,0]))
    #     print('max_value target_data: ', np.nanmax(target_data[:,0]))
        
    #     print('min_value all data: ', min_value)
    #     print('max_value all data: ', max_value)
    #     print('output.shape: ', output[:,0].shape)
    #     print('target_data.shape: ', target_data[:,0].shape)

    #     two_img('result/' + str(step).zfill(3), lon, lat, lon_min, lon_max, lat_min, lat_max,
    #             delta_lon, delta_lat, triangles, target_data[:,0], output[:,0], min_value, max_value, time)
    
    # save_data = np.stack([pred_list, target_list], axis=0)      # shape: (144, H, W)
    # print(save_data.shape)
    # np.savez_compressed('GGB_240512T00.npz', data=save_data)
    # print(dataset.shape)
    # for i in range(0, 144):
        # print(dataset[:, i, :, :].shape)

    # for i in range(0, 144):
    #     a = dataset_processing.reverse(dataset[i,:,:], 'dataset/scaler/GGB_240512T00_scaler.pkl')
    #     min_value = min(min_value, np.nanpercentile(a, 2))
    #     max_value = max(max_value, np.nanpercentile(a, 98))
    
    
    # a = dataset_processing.reverse(np.expand_dims(dataset[0,: , :, 5], axis=-1), 'dataset/scaler/GGB_240512T00_scaler_1day.pkl')
    # a = dataset_processing.reverse(np.expand_dims(dataset[0,: , :, 5], axis=-1), 'dataset/scaler/GGB_240512T00_scaler_1day.pkl')
    # a = dataset_processing.reverse(dataset[0,: , :, :], 'dataset/scaler/GGB_240512T00_scaler_1day.pkl')
    # b = dataset_processing.reverse(dataset[1,: , :, :], 'dataset/scaler/GGB_240512T00_scaler_1day.pkl')
    # min_value = min(min_value, np.nanpercentile(a[:, :, 5], 0.5))
    # min_value = min(min_value, np.nanpercentile(b[:, :, 5], 0.5))

    # max_value = max(max_value, np.nanpercentile(a[:, :, 5], 99.5))
    # max_value = max(max_value, np.nanpercentile(b[:, :, 5], 99.5))
    # print('min_value: ', min_value)
    # print('max_value: ', max_value)
    # for step in range(0, 144,6):
    #     output = a[step, :, 7]
    #     target_data = b[step, :, 7]
    #     min_value = min(np.nanmin(output), np.nanmin(target_data))
    #     max_value = max(np.nanmax(output), np.nanmax(target_data))
    #     print('step: ', step)
    #     print('min_value all data: ', min_value)
    #     print('max_value all data: ', max_value)

    #     print('min_value output: ', np.nanmin(output))
    #     print('max_value output: ', np.nanmax(output))

    #     print('min_value target_data: ', np.nanmin(target_data))
    #     print('max_value target_data: ', np.nanmax(target_data))

    #     two_img('result/' + str(step).zfill(3), lon, lat, lon_min, lon_max, lat_min, lat_max,
    #             delta_lon, delta_lat, triangles, target_data, output, min_value, max_value)
    

        