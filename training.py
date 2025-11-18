import numpy as np
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import elementtransformer
import os
import time

class FVCOMDataset(Dataset):
    def __init__(self, data_dir, total_timesteps=144*3, seq_len=6):
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.total_timesteps = total_timesteps
        self.start_indices = list(range(total_timesteps - seq_len))

    def __len__(self):
        return len(self.start_indices)

    def __getitem__(self, idx):
        start_t = self.start_indices[idx]
        
        inputs = []
        for i in range(self.seq_len):
            path = os.path.join(self.data_dir, f"step_{start_t + i:03d}.npz")
            frame = np.load(path)['data']
            inputs.append(frame)
        inputs = np.stack(inputs, axis=0)

        target_path = os.path.join(self.data_dir, f"step_{start_t + self.seq_len:03d}.npz")
        target = np.load(target_path)['data']
        target = np.expand_dims(target, axis=0)

        return torch.from_numpy(inputs).float(), torch.from_numpy(target).float()

    
def training_test():
    batch_size = 1
    num_nodes = 100
    t_in = 6
    t_out = 1
    var_in = 2
    var_out = 2
    embed_dim = 128

    x = torch.randn(batch_size, t_in, num_nodes, var_in)
    target = torch.randn(batch_size, t_out, num_nodes, var_out)

    model = elementtransformer.FVCOMModel(var_in=var_in,var_out=var_out,triangle=num_nodes,embed_dim=embed_dim)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    optimizer.zero_grad()

    output = model(x)
    print("Output shape:", output.shape)

    loss = criterion(output, target)
    print("Loss:", loss.item())

    loss.backward()

    grads = [p.grad for p in model.parameters() if p.grad is not None]
    print("All gradients non-zero?", all(g.abs().sum() > 1e-8 for g in grads))

    optimizer.step()

    print("End-to-end training step completed successfully!")

def training_u_v(data_dir, num_epochs, checkpoint_name_out):
    start_time = time.time()
    best_loss = float('inf')
    early_stop_cnt= 0
    best_epoch = 0

    batch_size = 1
    num_nodes = 115443
    t_in = 1
    t_out = 1
    var_in = 2
    var_out = 2
    embed_dim = 128
    device = "cuda" if torch.cuda.is_available() else "cpu"
    full_dataset = FVCOMDataset(
        data_dir=data_dir,
        total_timesteps=10,
        seq_len=1
    )
    total_samples = len(full_dataset)
    train_size = int(0.8 * total_samples)
    val_size = total_samples - train_size

    train_indices = list(range(train_size))
    val_indices = list(range(train_size, total_samples))

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)

    model = elementtransformer.FVCOMModel(var_in=var_in,var_out=var_out,triangle=num_nodes, embed_dim=embed_dim, t_in=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss().cuda()

    model.train()
    for epoch in range(num_epochs):
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            pred = model(x)
            print("epoch: ", epoch+1, " pred:   ", pred.shape)
            print("epoch: ", epoch+1, " target: ", y.shape)
            
            loss = criterion(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)

                pred = model(x)
                loss = criterion(pred, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                val_loss += loss.item()
                val_loss += loss.item()
        val_loss /= len(val_loader)
        model.train()
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, Best: {best_loss:.6f}")

        if val_loss < best_loss:

            best_loss = val_loss
            best_epoch = epoch + 1
            early_stop_cnt = 0

            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss,
                'loss': loss,
            }, checkpoint_name_out)

            print(f"Model saved at epoch {epoch+1} with val_loss: {val_loss:.6f}")
        else:
            early_stop_cnt += 1
            print(f"No improvement. Current val_loss: {val_loss:.6f}, Best so far: {best_loss:.6f}, Best epoch {best_epoch}")
        if early_stop_cnt > 25:
            break
    total_time = time.time() - start_time
    hours, rem = divmod(total_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"Total training time: {int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}")


if __name__ == "__main__":
    start_time = time.time()
    timestamp_str = time.strftime("%Y_%m_%d_%H_%M", time.localtime(start_time))
    training_u_v('dataset/step_data', 10, "checkpoints/" + timestamp_str + "_local_model.pth")