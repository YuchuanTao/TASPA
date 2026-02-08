import torch
import torch.nn as nn
import numpy as np

# =========================================================
# 1. Temporal Enhancement Module (Causal Convolution)
# =========================================================

class SingleCausalConv(nn.Module):
    """
    Single causal convolution block:
    Depthwise Conv1D + LayerNorm + Activation + Dropout

    Input : [B, T, C]
    Output: [B, T, C]
    """
    def __init__(self, num_sensors=14, kernel_size=3, dropout=0.1):
        super().__init__()
        self.kernel_size = kernel_size

        # Depthwise causal convolution
        self.conv = nn.Conv1d(
            in_channels=num_sensors,
            out_channels=num_sensors,
            kernel_size=kernel_size,
            groups=num_sensors
        )

        self.norm = nn.LayerNorm(num_sensors)
        self.activation = nn.Hardswish(inplace=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, T, C] → [B, C, T]
        x = x.permute(0, 2, 1)

        # Left padding to ensure causality
        pad_size = self.kernel_size - 1
        x = torch.nn.functional.pad(x, (pad_size, 0))

        # Causal convolution
        out = self.conv(x)  # [B, C, T]

        # Back to [B, T, C]
        out = out.permute(0, 2, 1)
        out = self.norm(out)
        out = self.activation(out)
        out = self.dropout(out)

        return out


class CausalConvBlock(nn.Module):
    """
    Two-layer causal convolution block with residual connection
    """
    def __init__(self, num_sensors=14, kernel_size=3, dropout=0.1):
        super().__init__()
        self.conv1 = SingleCausalConv(num_sensors, kernel_size, dropout)
        self.conv2 = SingleCausalConv(num_sensors, kernel_size, dropout)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return out + x


class TemporalEnhancementModule(nn.Module):
    """
    Stack of causal convolution blocks for temporal feature enhancement
    """
    def __init__(self, num_sensors, kernel_size, num_blocks, dropout):
        super().__init__()
        self.blocks = nn.Sequential(
            *[CausalConvBlock(num_sensors, kernel_size, dropout)
              for _ in range(num_blocks)]
        )

    def forward(self, x):
        return self.blocks(x)


# =========================================================
# 2. Temporal Multi-Head Attention with Prior Knowledge
# =========================================================

class TimeMultiHeadAttention(nn.Module):
    """
    Temporal multi-head attention with:
    - Causal constraint
    - Time-distance prior
    - Operating-condition similarity prior
    """
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0

        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.softmax = nn.Softmax(dim=-1)

        # Learnable scaling parameters for temporal prior
        self.eta = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(1.0))
        self.alpha = nn.Parameter(torch.tensor(1.0))

    def build_temporal_prior(self, seq_len, device, oc):
        """
        Construct temporal + operating-condition prior matrix

        oc: [B, T, 3]
        return: [B, 1, T, T]
        """
        B = oc.size(0)

        # Temporal distance |i - j|
        i = torch.arange(seq_len, device=device).view(seq_len, 1)
        j = torch.arange(seq_len, device=device).view(1, seq_len)
        dist_t = torch.abs(i - j).float()

        # Operating-condition similarity
        oc_i = oc.unsqueeze(2)
        oc_j = oc.unsqueeze(1)
        oc_dist = torch.sum((oc_i - oc_j) ** 2, dim=-1)
        oc_sim = torch.exp(-oc_dist)

        # Temporal decay
        time_decay = self.eta / (1.0 + dist_t.unsqueeze(0) ** torch.abs(self.beta))

        # Combined prior
        D = self.alpha * time_decay * oc_sim
        return D.unsqueeze(1)

    def causal_mask(self, seq_len, device):
        """
        Prevent attention to future time steps
        """
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask.unsqueeze(0).unsqueeze(0)

    def forward(self, x, oc):
        """
        x : [B, T, d_model]
        oc: [B, T, 3]
        """
        B, T, _ = x.size()

        Q = self.W_Q(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_K(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_V(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)

        # Apply temporal + OC prior
        D = self.build_temporal_prior(T, x.device, oc)
        scores = scores * D

        # Apply causal constraint
        scores = scores + self.causal_mask(T, x.device)

        attn = self.softmax(scores)
        out = torch.matmul(attn, V)

        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.out_proj(out)


# =========================================================
# 3. Sensor Multi-Head Attention with Distance Prior
# =========================================================

class SensorMultiHeadAttention(nn.Module):
    """
    Sensor-wise attention guided by a predefined sensor distance matrix
    """
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0

        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.softmax = nn.Softmax(dim=-1)

        self.scale = nn.Parameter(torch.tensor(1.0))
        self.power = nn.Parameter(torch.tensor(1.0))

        # Predefined sensor distance matrix (14 × 14)
        self.register_buffer(
            "sensor_distance",
            torch.tensor(np.array([
                [0, 1, 4, 1, 1, 2, 1, 2, 1, 3, 2, 1, 3, 5],
                [1, 0, 3, 1, 2, 1, 2, 1, 2, 2, 3, 2, 2, 4],
                [4, 3, 0, 3, 5, 4, 3, 2, 5, 5, 6, 5, 1, 1],
                [1, 1, 3, 0, 2, 2, 1, 1, 2, 3, 3, 2, 2, 4],
                [1, 2, 5, 2, 0, 3, 2, 3, 1, 4, 1, 2, 4, 6],
                [2, 1, 4, 2, 3, 0, 3, 2, 3, 1, 4, 3, 3, 5],
                [1, 2, 3, 1, 2, 3, 0, 1, 2, 4, 3, 2, 2, 4],
                [2, 1, 2, 1, 3, 2, 1, 0, 3, 3, 4, 3, 1, 3],
                [1, 2, 5, 2, 1, 3, 2, 3, 0, 4, 2, 2, 4, 6],
                [3, 2, 5, 3, 4, 1, 4, 3, 4, 0, 5, 4, 4, 6],
                [2, 3, 6, 3, 1, 4, 3, 4, 2, 5, 0, 3, 5, 7],
                [1, 2, 5, 2, 2, 3, 2, 3, 2, 4, 3, 0, 4, 6],
                [3, 2, 1, 2, 4, 3, 2, 1, 4, 4, 5, 4, 0, 2],
                [5, 4, 1, 4, 6, 5, 4, 3, 6, 6, 7, 6, 2, 0]
            ])).float()
        )

    def build_sensor_prior(self, num_sensors, device):
        D = self.sensor_distance[:num_sensors, :num_sensors].to(device)
        return (self.scale / (1.0 + D ** self.power)).unsqueeze(0).unsqueeze(0)

    def forward(self, x):
        """
        x: [B, N, d_model]
        """
        B, N, _ = x.size()

        Q = self.W_Q(x).view(B, N, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_K(x).view(B, N, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_V(x).view(B, N, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        scores = scores * self.build_sensor_prior(N, x.device)

        attn = self.softmax(scores)
        out = torch.matmul(attn, V)

        out = out.transpose(1, 2).contiguous().view(B, N, -1)
        return self.out_proj(out)


# =========================================================
# 4. Time Encoder Layer & Stack
# =========================================================

class TimeEncoderLayer(nn.Module):
    """
    Single layer of time encoder:
    - Temporal multi-head attention with prior
    - Residual + LayerNorm
    - Feedforward network with residual
    """
    def __init__(self, d_model, num_heads, dim_ff, dropout=0.1):
        super().__init__()
        self.mha = TimeMultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.Hardswish(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, oc):
        # Temporal attention with residual + normalization
        x = self.norm1(x + self.dropout(self.mha(x, oc)))
        # Feedforward with residual + normalization
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


class TimeEncoder(nn.Module):
    """
    Stack of TimeEncoderLayer
    """
    def __init__(self, d_model, num_heads, dim_ff, num_layers, dropout):
        super().__init__()
        self.layers = nn.ModuleList([
            TimeEncoderLayer(d_model, num_heads, dim_ff, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, oc):
        for layer in self.layers:
            x = layer(x, oc)
        return x  # [B, T, d_model]


# =========================================================
# 5. Sensor Encoder Layer & Stack
# =========================================================

class SensorEncoderLayer(nn.Module):
    """
    Single layer of sensor encoder:
    - Sensor multi-head attention guided by sensor distance prior
    - Residual + LayerNorm
    - Feedforward network with residual
    """
    def __init__(self, d_model, num_heads, dim_ff, dropout=0.1):
        super().__init__()
        self.mha = SensorMultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.Hardswish(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.norm1(x + self.dropout(self.mha(x)))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


class SensorEncoder(nn.Module):
    """
    Stack of SensorEncoderLayer
    """
    def __init__(self, d_model, num_heads, dim_ff, num_layers, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            SensorEncoderLayer(d_model, num_heads, dim_ff, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x  # [B, N, d_model]


# =========================================================
# 6. LICA: Cross-channel SE fusion for time & sensor features
# =========================================================

class LICA(nn.Module):
    """
    Cross-channel attention module for fusing temporal and sensor features
    - Implements ECA-style attention on flattened features
    - Outputs final RUL prediction
    """
    def __init__(self, num_time_steps, num_sensors, eca_kernel_size, d_model, hidden_dim):
        super().__init__()
        self.num_time_steps = num_time_steps
        self.num_sensors = num_sensors
        self.total_features = num_time_steps + num_sensors

        # ECA 1D convolution on fused channels
        self.eca_conv = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=eca_kernel_size,
            padding=eca_kernel_size // 2,
            bias=False
        )
        self.sigmoid = nn.Sigmoid()

        # Fully connected layers for RUL regression
        self.fused_dim = self.total_features * d_model
        self.fc_hidden = nn.Sequential(
            nn.Linear(self.fused_dim, hidden_dim),
            nn.Hardswish(inplace=True)
        )
        self.fc_rul = nn.Linear(hidden_dim, 1)

    def forward(self, time_feat, sensor_feat):
        """
        time_feat  : [B, T, d_model]
        sensor_feat: [B, N, d_model]
        """
        # 1) Squeeze each feature map along embedding dimension
        time_squeezed = time_feat.mean(dim=-1)    # [B, T]
        sensor_squeezed = sensor_feat.mean(dim=-1)  # [B, N]

        # 2) Concatenate along channel dimension
        combined = torch.cat([time_squeezed, sensor_squeezed], dim=-1)  # [B, T+N]

        # 3) Apply ECA attention
        attn = self.eca_conv(combined.unsqueeze(1))  # [B, 1, T+N]
        attn = self.sigmoid(attn).squeeze(1)         # [B, T+N]

        # 4) Split weights for time and sensor
        time_weight = attn[:, :self.num_time_steps].unsqueeze(-1)  # [B, T, 1]
        sensor_weight = attn[:, self.num_time_steps:].unsqueeze(-1)  # [B, N, 1]

        # 5) Reweight features (residual connection)
        weighted_time = time_feat * time_weight + time_feat
        weighted_sensor = sensor_feat * sensor_weight + sensor_feat

        # 6) Flatten and fully connect
        fused_flat = torch.cat([weighted_time, weighted_sensor], dim=1).view(weighted_time.size(0), -1)
        hidden = self.fc_hidden(fused_flat)
        rul_pred = self.fc_rul(hidden).squeeze(-1)
        return rul_pred


# =========================================================
# 7. TASPA: Full model integrating TEM, Attention Encoders, and LICA
# =========================================================

class TASPA(nn.Module):
    """
    TASPA model for RUL prediction:
    - Temporal Enhancement Module (TEM)
    - Time & Sensor Multi-Head Attention Encoders
    - Cross-channel fusion (LICA) for final RUL
    """
    def __init__(self,
                 tem_kernel_size, tem_num_blocks,
                 num_time_steps, num_sensors,
                 attention_d_model, attention_num_heads, attention_dim_ff, attention_num_layers,
                 eca_kernel_size, fused_hidden, dropout):

        super().__init__()

        # Temporal convolution module
        self.tem = TemporalEnhancementModule(num_sensors, tem_kernel_size, tem_num_blocks, dropout)

        # Linear embedding for attention
        self.sensor_embedding = nn.Linear(num_sensors, attention_d_model)
        self.time_embedding = nn.Linear(num_time_steps, attention_d_model)

        # Encoders
        self.time_encoder = TimeEncoder(attention_d_model, attention_num_heads, attention_dim_ff, attention_num_layers, dropout)
        self.sensor_encoder = SensorEncoder(attention_d_model, attention_num_heads, attention_dim_ff, attention_num_layers, dropout)

        # Fusion & regression
        self.fusion = LICA(num_time_steps, num_sensors, eca_kernel_size, attention_d_model, fused_hidden)

    def forward(self, x, oc):
        """
        x : [B, T, N] sensor sequence
        oc: [B, T, 3] operating conditions
        """
        # 1) Temporal enhancement
        enhanced = self.tem(x)

        # 2) Embedding for attention
        sensor_emb = self.sensor_embedding(enhanced)           # [B, T, d_model]
        time_emb = self.time_embedding(enhanced.permute(0, 2, 1))  # [B, N, d_model]

        # 3) Encoders
        time_out = self.time_encoder(sensor_emb, oc)   # [B, T, d_model]
        sensor_out = self.sensor_encoder(time_emb)     # [B, N, d_model]

        # 4) Fusion and RUL prediction
        rul_pred = self.fusion(time_out, sensor_out)
        return rul_pred
