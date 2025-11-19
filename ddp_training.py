import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Subset
from torch.utils.data.distributed import DistributedSampler
import time
import os
import numpy as np
import elementtransformer

def train_zero_epoch_ddp(data_dir, num_epochs, checkpoint_name_out):
    start_time = time.time()
    best_loss = float('inf')
    early_stop_cnt= 0
    best_epoch = 0
    batch_size = 1
    num_nodes = 115443
    t_in = 1
    t_out = 1
    var_in = 10
    var_out = 10
    embed_dim = 256
    device = "cuda" if torch.cuda.is_available() else "cpu"

    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group("nccl", rank=local_rank, world_size=world_size)
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(local_rank)
    model = elementtransformer.FVCOMModel(var_in=var_in,var_out=var_out,
                                          triangle=num_nodes, embed_dim=embed_dim,
                                          t_in=t_in).to(device)
    model = DDP(model, device_ids=[local_rank])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = elementtransformer.WeightedMAEMSELoss().cuda()
    
    full_dataset = elementtransformer.FVCOMDataset(
        data_dir=data_dir,
        total_timesteps=12,
        pred_step=t_in
    )

    total_samples = len(full_dataset)
    train_size = int(0.8 * total_samples)
    val_size = total_samples - train_size

    train_indices = list(range(train_size))
    val_indices = list(range(train_size, total_samples))

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=local_rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=local_rank, shuffle=False)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, sampler=train_sampler,
        num_workers=2, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, sampler=val_sampler,
        num_workers=2, pin_memory=True
    )

    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        train_loss = 0.0
        for inp, target in train_loader:
            inp = inp.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            pred = model(inp)
            loss = criterion(pred, target)
            print("epoch: ", epoch+1, " input:  ", inp.shape)
            print("epoch: ", epoch+1, " pred:   ", pred.shape)
            print("epoch: ", epoch+1, " target: ", target.shape)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inp, target in val_loader:
                inp = inp.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)

                pred = model(inp)
                loss = criterion(pred, target)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        model.train()

        val_loss_tensor = torch.tensor(val_loss, device=device)
        dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
        val_loss = val_loss_tensor.item() / world_size

        if local_rank == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, Best: {best_loss:.6f}")

            if val_loss < best_loss:
                best_loss = val_loss
                best_epoch = epoch + 1
                early_stop_cnt = 0

                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'train_loss': train_loss,
                }, checkpoint_name_out)

                print(f"Model saved at epoch {epoch+1} with val_loss: {val_loss:.6f}")
            else:
                early_stop_cnt += 1
                print(f"No improvement. Current val_loss: {val_loss:.6f}, Best so far: {best_loss:.6f}, Best epoch {best_epoch}")

            if early_stop_cnt > 25:
                print("Early stopped.")
                break

    total_time = time.time() - start_time
    if local_rank == 0:
        hours, rem = divmod(total_time, 3600)
        minutes, seconds = divmod(rem, 60)
        print(f"Total training time: {int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}")

    dist.destroy_process_group()


def main():
    world_size = torch.cuda.device_count()
    print(world_size, " GPU found")
    assert world_size > 0, "No GPUs available"
    start_time = time.time()
    timestamp_str = time.strftime("%Y_%m_%d_%H_%M", time.localtime(start_time))
    train_zero_epoch_ddp('dataset/step_data', 100, "checkpoints/" + timestamp_str + '_' + str(world_size) + "_Inha_GPU_Server_model.pth")

if __name__ == "__main__":
    main()