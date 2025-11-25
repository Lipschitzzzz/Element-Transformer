import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import os

class FVCOMDataset(Dataset):
    def __init__(self, data_dir, total_timesteps=144*7, steps_per_file=144, pred_step=1):
        self.data_dir = data_dir
        self.file_list = os.listdir(data_dir)
        print(self.file_list)
        self.total_timesteps = total_timesteps
        self.steps_per_file = steps_per_file
        self.pred_step = pred_step
        
        self.max_start_t = total_timesteps - pred_step - 1
        if self.max_start_t < 0:
            raise ValueError(f"pred_step={pred_step} too large for total_timesteps={total_timesteps}")
        
        self.total_samples = self.max_start_t + 1

    def _global_to_local(self, global_t):
        file_idx = global_t // self.steps_per_file
        local_t = global_t % self.steps_per_file
        return file_idx, local_t

    def _load_frame(self, global_t):
        file_idx, local_t = self._global_to_local(global_t)
        path = os.path.join(self.data_dir, self.file_list[file_idx])
        # print(self.data_dir + self.file_list[file_idx])
        data = np.load(path)
        return data[local_t]

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        t = idx
        t_target = t + self.pred_step

        input_frame = self._load_frame(t)
        target_frame = self._load_frame(t_target)

        input_tensor = torch.from_numpy(input_frame).float().unsqueeze(0)
        target_tensor = torch.from_numpy(target_frame).float().unsqueeze(0)

        return input_tensor, target_tensor

class WeightedMAEMSELoss(nn.Module):
    def __init__(self, weight_mae=1.0, weight_mse=0.2):
        super().__init__()
        self.weight_mae = weight_mae
        self.weight_mse = weight_mse


        channel_weights = torch.ones(18)
        channel_weights[0:15] = 1.0
        channel_weights[15:18] = 5.0
        self.register_buffer('channel_weights', channel_weights)

    def forward(self, pred, target):
        weights = self.channel_weights.view([1] * (pred.dim() - 1) + [-1])
        # print(pred.shape)
        # print(target.shape)
        # print(weights.shape)
        abs_error = torch.abs(pred - target)
        squared_error = (pred - target) ** 2
        weighted_abs_error = weights * abs_error
        weighted_squared_error = weights * squared_error
        mae = weighted_abs_error.mean()
        mse = weighted_squared_error.mean()
        loss = self.weight_mae * mae + self.weight_mse * mse
        return loss
    
class NodeEmbedding(nn.Module):
    def __init__(self, t_in=6, in_chans=4, embed_dim=768):
        super().__init__()
        self.embed_dim = embed_dim
        self.in_chans = in_chans
        self.t_in = t_in

    def forward(self, x):
        B, T, C, P = x.shape
        x = x.reshape(B * T, C, P)
        x = x.flatten(2).transpose(1, 2)
        x = x.reshape(B, T, self.n_patches_total, self.embed_dim)
        return x
    
class TriangleEmbedding(nn.Module):
    def __init__(self, t_in=6, in_chans=4, embed_dim=768):
        super().__init__()
        self.embed_dim = embed_dim
        self.in_chans = in_chans
        self.t_in = t_in
        
    def forward(self, x):
        
        B, T, C, P = x.shape

        x = x.reshape(B * T, C, P)

        x = x.flatten(2).transpose(1, 2)
        x = x.reshape(B, T, self.n_patches_total, self.embed_dim)

        return x
    
class NeighborSparseAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, neighbor_table, dropout=0.1):
        """
        neighbor_table: numpy array or torch tensor of shape (K, N)
                        where K=3, N=seq_len.
                        Each column i lists up to K neighbor indices for token i.
                        Use -1 to indicate "no neighbor".
        """
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        if isinstance(neighbor_table, np.ndarray):
            neighbor_table = torch.from_numpy(neighbor_table).long()
        else:
            neighbor_table = neighbor_table.long()

        K, N = neighbor_table.shape
        self.K = K
        self.N = N

        valid_mask = neighbor_table != -1  # (K, N), bool
        neighbor_table = neighbor_table.clamp(min=0)

        self.register_buffer("neighbor_indices", neighbor_table)
        self.register_buffer("valid_mask", valid_mask.float())

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        assert N == self.N, f"Input sequence length {N} != expected {self.N}"

        # Project Q, K, V
        q = self.q_proj(x)  # (B, N, C)
        k = self.k_proj(x)  # (B, N, C)
        v = self.v_proj(x)  # (B, N, C)

        # Reshape for multi-head: (B, N, H, D) -> (B, H, N, D)
        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, N, D)
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, N, D)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, N, D)

        # Gather neighbors for K and V: for each position i, get its K neighbors
        # neighbor_indices: (K, N) -> expand to (B, H, K, N)
        idx = self.neighbor_indices.unsqueeze(0).unsqueeze(0)  # (1, 1, K, N)
        idx = idx.expand(B, self.num_heads, -1, -1)  # (B, H, K, N)

        # k: (B, H, N, D) -> gather over N dimension using idx -> (B, H, K, N, D)
        k_neighbors = torch.gather(k.unsqueeze(2).expand(-1, -1, self.K, -1, -1), 3, idx.unsqueeze(-1).expand(-1, -1, -1, -1, self.head_dim))
        v_neighbors = torch.gather(v.unsqueeze(2).expand(-1, -1, self.K, -1, -1), 3, idx.unsqueeze(-1).expand(-1, -1, -1, -1, self.head_dim))

        # q for each position: (B, H, N, D) -> unsqueeze to (B, H, 1, N, D)
        q_exp = q.unsqueeze(2)  # (B, H, 1, N, D)

        # Compute attention scores: (B, H, K, N)
        attn_scores = (q_exp * k_neighbors).sum(dim=-1) * self.scale  # (B, H, K, N)

        # Apply validity mask: invalid neighbors get -inf
        valid_mask = self.valid_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, K, N)
        attn_scores = attn_scores.masked_fill(valid_mask == 0, float('-inf'))

        # Softmax over K dimension (only among valid neighbors)
        attn_weights = attn_scores.softmax(dim=2)  # (B, H, K, N)
        attn_weights = self.attn_drop(attn_weights)

        # Weighted sum of V neighbors
        out = (attn_weights.unsqueeze(-1) * v_neighbors).sum(dim=2)  # (B, H, N, D)

        # Reshape back
        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out
    
class SparseTransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio, neighbor_table, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = NeighborSparseAttention(embed_dim, num_heads, neighbor_table, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class Encoder(nn.Module):
    def __init__(self, node=60882, triangle=10, var_in=2, embed_dim=768, depth=12, t_in=6, num_heads=12, mlp_ratio=4., neighbor_table=None, dropout=0.1, num_layers=2):
        super().__init__()
        self.var_in = var_in
        self.node = node
        self.triangle = triangle
        self.t_in = t_in
        self.embed_dim = embed_dim
        self.embedding_layer = nn.Linear(var_in, embed_dim)
        self.spatial_pos_embed = nn.Parameter(torch.zeros(1, triangle, embed_dim))
        assert neighbor_table.shape == (3, self.triangle), f"neighbor_table must be (3, {self.triangle})"
        self.pos_drop = nn.Dropout(p=dropout)

        self.transformer_blocks = nn.ModuleList([
            SparseTransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                neighbor_table=neighbor_table,
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # encoder_layer = nn.TransformerEncoderLayer(
        #     d_model=embed_dim,
        #     nhead=2,
        #     dim_feedforward=int(embed_dim * mlp_ratio),
        #     dropout=dropout,
        #     activation='gelu',
        #     batch_first=True
        # )
        # self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.spatial_pos_embed, std=0.02)
        self.apply(self._init_layer_weights)

    def _init_layer_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        B, T, E, C = x.shape
        assert E == self.triangle, f"Expected {self.triangle} elements, got {E}"
        assert C == self.var_in, f"Expected {self.var_in} features, got {C}"
        x = self.embedding_layer(x)
        x = x + self.spatial_pos_embed.unsqueeze(1)
        x = self.pos_drop(x)

        x = x.view(B, T * E, self.embed_dim)

        for blk in self.transformer_blocks:
            x = blk(x)
        x = self.norm(x)
        return x

class Decoder(nn.Module):
    def __init__(self, node=1, triangle=115443, embed_dim=768, var_in=1, var_out=1, t_in=6, t_out=1):
        super().__init__()
        self.triangle = triangle
        self.t_in = t_in
        self.t_out = t_out
        self.var_in = var_in
        self.var_out = var_out
        self.embed_dim = embed_dim

        self.proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, t_out * var_out)
        )

    def forward(self, x):
        B, C, E = x.shape
        assert C == self.t_in * self.triangle, f"Expected seq_len={self.t_in * self.triangle}, got {C}"
        x = x.view(B, self.t_in, self.triangle, E)
        x = x[:, -1, :, :]
        x = self.proj(x)
        x = x.view(B, self.triangle, self.t_out, self.var_out)
        x = x.permute(0, 2, 1, 3)
        return x

class ElementTransformerNet(nn.Module):
    def __init__(self, var_in=2, var_out=4, t_in=6, t_out=1, embed_dim=768, triangle=100,
                 depth=12, num_heads=12, neighbor_table=None, mlp_ratio=4., dropout=0.1):
        super().__init__()
        self.t_in = t_in
        self.t_out = t_out
        self.var_out = var_out
        
        self.encoder = Encoder(
            var_in=var_in,
            embed_dim=embed_dim,
            triangle=triangle,
            depth=depth,
            t_in = t_in,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            neighbor_table=neighbor_table,
            dropout=dropout
        )
        self.decoder = Decoder(
            embed_dim=embed_dim,
            var_out=var_out,
            t_in=t_in,
            t_out=t_out
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def predict(self, checkpoint_name, input_data):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(checkpoint_name, map_location=device)

        self.load_state_dict(checkpoint['model_state_dict'])

        self.eval()
        with torch.no_grad():
            output = self(input_data)

        # prediction = output.squeeze(0).cpu().numpy()
        
        return output

def FVCOMModel(var_in=2, var_out=1, t_in=6, t_out=1, triangle=100, embed_dim=256,
             depth=2,num_heads=2,mlp_ratio=4,neighbor_table=None,dropout=0.1):
    
    model = ElementTransformerNet(var_in=var_in,var_out=var_out,t_in=t_in,triangle=triangle,
                                  t_out=t_out,embed_dim=embed_dim,depth=depth,
                                  num_heads=num_heads, neighbor_table=neighbor_table,)
    return model

if __name__ == "__main__":
    pass
    