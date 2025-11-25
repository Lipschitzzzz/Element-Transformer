import numpy as np
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import elementtransformer, visualization
import time

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
    var_in = 18
    var_out = 18
    embed_dim = 256
    nbe = np.load('nbe.npy')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    full_dataset = elementtransformer.FVCOMDataset(
        data_dir=data_dir,
        total_timesteps=3,
        pred_step=t_in
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

    model = elementtransformer.FVCOMModel(var_in=var_in,var_out=var_out,
                                          triangle=num_nodes, embed_dim=embed_dim,
                                          t_in=t_in,neighbor_table=nbe).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = elementtransformer.WeightedMAEMSELoss().cuda()
    
    visualization.count_parameters(model)

    model.train()
    for epoch in range(num_epochs):
        train_loss = 0.0
        for tx, ty in train_loader:
            tx, ty = tx.to(device), ty.to(device)

            pred = model(tx)
            print("epoch: ", epoch+1, " input:  ", tx.shape)
            print("epoch: ", epoch+1, " pred:   ", pred.shape)
            print("epoch: ", epoch+1, " target: ", ty.shape)
            
            loss = criterion(pred, ty)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for vx, vy in val_loader:
                vx, vy = vx.to(device), vy.to(device)

                pred = model(vx)
                loss = criterion(pred, vy)

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
    training_u_v('dataset/step_data', 10, "checkpoints/" + timestamp_str + "_Inha_GPU_Server.pth")