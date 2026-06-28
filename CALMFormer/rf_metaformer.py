"""
RF-MetaFormer: MetaFormer for Few-shot Specific Emitter Identification
Architecture: PatchEmbedding -> N x MetaFormerBlock -> GlobalPooling -> Feature Vector
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
# Token Mixers
# ─────────────────────────────────────────────

class IdentityMixer(nn.Module):
    """不做任何token mixing，验证MetaFormer框架本身的作用"""
    def forward(self, x):
        return x


class PoolingMixer(nn.Module):
    """
    用局部平均池化做token mixing（PoolFormer思路）
    物理直觉：平滑局部时域波动，保留稳定的硬件指纹
    """
    def __init__(self, pool_size=3):
        super().__init__()
        # avg_pool - identity = 局部残差，避免信息损失
        self.pool = nn.AvgPool1d(pool_size, stride=1, padding=pool_size // 2, count_include_pad=False)

    def forward(self, x):
        # x: [B, N, D] -> transpose -> pool -> transpose
        x = x.transpose(1, 2)          # [B, D, N]
        x = self.pool(x) - x           # 局部残差（PoolFormer原文设计）
        return x.transpose(1, 2)       # [B, N, D]


class ConvMixer(nn.Module):
    """
    Depthwise Conv1d做token mixing
    物理直觉：捕获局部时域畸变（IQ不平衡、PA非线性）
    """
    def __init__(self, dim, kernel_size=7):
        super().__init__()
        self.dw_conv = nn.Conv1d(
            dim, dim, kernel_size,
            padding=kernel_size // 2, groups=dim, bias=False
        )

    def forward(self, x):
        x = x.transpose(1, 2)          # [B, D, N]
        x = self.dw_conv(x)
        return x.transpose(1, 2)       # [B, N, D]


class SpectralMixer(nn.Module):
    """
    FFT频域token mixing
    物理直觉：捕获频谱指纹、CFO、相位噪声等全局频域特征
    """
    def __init__(self, num_tokens):
        super().__init__()
        # 可学习的复数频域滤波权重，初始化为接近恒等滤波。
        weight = torch.zeros(num_tokens // 2 + 1, 2, dtype=torch.float32)
        weight[:, 0] = 1.0
        weight = weight + torch.randn_like(weight) * 0.02
        self.complex_weight = nn.Parameter(weight)

    def forward(self, x):
        # x: [B, N, D]
        x_freq = torch.fft.rfft(x, dim=1, norm='ortho')           # [B, N//2+1, D]
        weight = torch.view_as_complex(self.complex_weight).unsqueeze(-1)  # [N//2+1, 1]
        x_freq = x_freq * weight
        x = torch.fft.irfft(x_freq, n=x.shape[1], dim=1, norm='ortho')
        return x                                                    # [B, N, D]


class AttentionMixer(nn.Module):
    """
    标准多头自注意力token mixing（Transformer-style baseline）
    用于检验全局内容自适应token交互是否比RF局部时域归纳偏置更有效。
    """
    def __init__(self, dim, num_heads=4, drop=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=drop,
            batch_first=True,
        )

    def forward(self, x):
        # x: [B, N, D]
        x, _ = self.attn(x, x, x, need_weights=False)
        return x


class SelectiveSSM(nn.Module):
    """
    单向选择性状态空间核（Mamba S6），纯PyTorch顺序扫描实现，无外部CUDA核依赖。
    输入/输出: [B, N, d_inner]
    """
    def __init__(self, d_inner, d_state=16, dt_rank=None, d_conv=4):
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state
        self.dt_rank = dt_rank or max(1, math.ceil(d_inner / 16))

        # 序列方向的因果深度卷积（Mamba中的短卷积）
        self.conv = nn.Conv1d(
            d_inner, d_inner, d_conv,
            padding=d_conv - 1, groups=d_inner, bias=True,
        )
        # 输入相关的 (Δ, B, C)
        self.x_proj = nn.Linear(d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, d_inner, bias=True)
        # 离散化前的连续 A（保证负实部，HiPPO式初始化为 1..d_state）
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_inner))
        # 让初始 Δ≈0.1，稳定早期训练（inverse-softplus）
        with torch.no_grad():
            self.dt_proj.bias.fill_(math.log(math.expm1(0.1)))

    def forward(self, x):
        # x: [B, N, d_inner]
        B, N, _ = x.shape
        xc = self.conv(x.transpose(1, 2))[..., :N].transpose(1, 2)  # 因果卷积 [B, N, d_inner]
        xc = F.silu(xc)

        x_dbl = self.x_proj(xc)                                     # [B, N, dt_rank+2*d_state]
        dt, Bm, Cm = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        dt = F.softplus(self.dt_proj(dt))                          # [B, N, d_inner]
        A = -torch.exp(self.A_log)                                # [d_inner, d_state]

        dA = torch.exp(dt.unsqueeze(-1) * A)                       # [B, N, d_inner, d_state]
        dBx = dt.unsqueeze(-1) * Bm.unsqueeze(2) * xc.unsqueeze(-1)  # [B, N, d_inner, d_state]

        h = torch.zeros(B, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(N):
            h = dA[:, t] * h + dBx[:, t]                           # [B, d_inner, d_state]
            ys.append((h * Cm[:, t].unsqueeze(1)).sum(-1))        # [B, d_inner]
        y = torch.stack(ys, dim=1)                                # [B, N, d_inner]
        return y + xc * self.D


class MambaMixer(nn.Module):
    """
    双向选择性SSM（Mamba）token mixer，纯PyTorch实现，无 mamba-ssm 依赖。
    物理直觉：线性复杂度的选择性长程时序建模，捕获贯穿整段信号的器件持久指纹。
    注意：此为「无RF改造」的 gate 基线版，仅检验长程建模能否破高-support瓶颈。
    """
    def __init__(self, dim, d_state=16, expand=2, d_conv=4):
        super().__init__()
        d_inner = expand * dim
        self.in_proj = nn.Linear(dim, 2 * d_inner, bias=False)
        self.ssm_fwd = SelectiveSSM(d_inner, d_state=d_state, d_conv=d_conv)
        self.ssm_bwd = SelectiveSSM(d_inner, d_state=d_state, d_conv=d_conv)
        self.out_proj = nn.Linear(d_inner, dim, bias=False)

    def forward(self, x):
        # x: [B, N, D]
        x_in, z = self.in_proj(x).chunk(2, dim=-1)   # 各 [B, N, d_inner]
        y_fwd = self.ssm_fwd(x_in)                    # 正向扫描
        y_bwd = self.ssm_bwd(x_in.flip(1)).flip(1)    # 反向扫描（IQ片段无因果方向）
        y = (y_fwd + y_bwd) * F.silu(z)               # 门控
        return self.out_proj(y)                       # [B, N, D]


# ─────────────────────────────────────────────
# RF-PMLS Mixer（相位感知多尺度局部-频谱混合，版本A：I/Q-only）
# ─────────────────────────────────────────────

class MultiScaleLocalBranch(nn.Module):
    """
    多尺度局部时域分支 M_l：多个不同 kernel 的 depthwise conv，
    捕获不同时间尺度上的 RF 指纹（短瞬态畸变 + 较长包络变化）。
    """
    def __init__(self, dim, kernel_sizes=(3, 7, 15)):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(dim, dim, k, padding=k // 2, groups=dim, bias=False)
            for k in kernel_sizes
        ])
        self.scale_weight = nn.Parameter(torch.ones(len(kernel_sizes)))

    def forward(self, x):
        xt = x.transpose(1, 2)                         # [B, D, N]
        w = torch.softmax(self.scale_weight, dim=0)
        out = sum(wi * conv(xt) for wi, conv in zip(w, self.convs))
        return out.transpose(1, 2)                     # [B, N, D]


class SpectralBranch(nn.Module):
    """频谱分支 M_s：可学习复数谱滤波，捕获频谱指纹/CFO/相噪等全局频域结构。"""
    def __init__(self, num_tokens):
        super().__init__()
        weight = torch.zeros(num_tokens // 2 + 1, 2, dtype=torch.float32)
        weight[:, 0] = 1.0
        weight = weight + torch.randn_like(weight) * 0.02
        self.complex_weight = nn.Parameter(weight)

    def forward(self, x):
        x_freq = torch.fft.rfft(x, dim=1, norm='ortho')
        weight = torch.view_as_complex(self.complex_weight).unsqueeze(-1)
        x_freq = x_freq * weight
        return torch.fft.irfft(x_freq, n=x.shape[1], dim=1, norm='ortho')


class PhaseAwareBranch(nn.Module):
    """
    相位感知分支 M_p（版本A：I/Q-only）：用相邻 token 的一阶/二阶差分
    近似相位动态（瞬时频率、相位漂移）。不显式输入 A/Δφ 通道，从 token
    表征中重构，保证与其他 mixer 对 raw-IQ 输入的公平对比。
    """
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x):
        # x: [B, N, D]
        dx = torch.zeros_like(x)
        dx[:, 1:] = x[:, 1:] - x[:, :-1]               # 一阶差分 ≈ 瞬时频率
        d2x = torch.zeros_like(x)
        d2x[:, 1:] = dx[:, 1:] - dx[:, :-1]            # 二阶差分 ≈ 相位曲率
        feat = torch.cat([x, dx, d2x], dim=-1)         # [B, N, 3D]
        return self.proj(feat)                          # [B, N, D]


class RFPMLSMixer(nn.Module):
    """
    RF-PMLS: Phase-aware Multi-scale Local-Spectral token mixer（版本A，I/Q-only）。
        M(H) = G_l ⊙ M_l(H) + G_s ⊙ M_s(H) + G_p ⊙ M_p(H)
    三分支（多尺度局部时域 / 可学习频谱 / 相位动态）+ SE 风格自适应门控融合。
    物理直觉：RF 硬件指纹分布在局部时域畸变、频域选择性、相位动态三类结构上，
    用与之对齐的归纳偏置组合，而非通用全局注意力。
    """
    def __init__(self, dim, num_tokens, kernel_sizes=(3, 7, 15)):
        super().__init__()
        self.local = MultiScaleLocalBranch(dim, kernel_sizes)
        self.spectral = SpectralBranch(num_tokens)
        self.phase = PhaseAwareBranch(dim)
        self.gate = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, dim * 3),
        )

    def forward(self, x):
        # x: [B, N, D]
        ml = self.local(x)
        ms = self.spectral(x)
        mp = self.phase(x)
        ctx = x.mean(dim=1)                            # [B, D] 全局上下文
        gl, gs, gp = self.gate(ctx).chunk(3, dim=-1)   # 各 [B, D]
        gl = torch.sigmoid(gl).unsqueeze(1)
        gs = torch.sigmoid(gs).unsqueeze(1)
        gp = torch.sigmoid(gp).unsqueeze(1)
        return gl * ml + gs * ms + gp * mp             # [B, N, D]


class RFLocalStatistics(nn.Module):
    """Patch-level RF statistics from raw I/Q samples for RF-aware token mixing."""
    def __init__(self, patch_size, lags=(1, 2, 4, 8, 16), eps=1e-6):
        super().__init__()
        self.patch_size = patch_size
        self.lags = tuple(k for k in lags if k < patch_size)
        self.eps = eps
        self.stat_dim = 2 + len(self.lags) + 4

    def forward(self, x):
        # x: [B, 2, L] -> patches: [B, 2, N, P]
        patches = x.unfold(dimension=-1, size=self.patch_size, step=self.patch_size)
        i = patches[:, 0]
        q = patches[:, 1]
        power = i.square() + q.square()

        energy = power.mean(dim=-1, keepdim=True)
        amp_var = power.var(dim=-1, unbiased=False, keepdim=True)

        corr_features = []
        mag = torch.sqrt(power + self.eps)
        for lag in self.lags:
            i0, q0 = i[..., lag:], q[..., lag:]
            i1, q1 = i[..., :-lag], q[..., :-lag]
            denom = mag[..., lag:] * mag[..., :-lag] + self.eps
            corr_re = (i0 * i1 + q0 * q1) / denom
            corr_im = (q0 * i1 - i0 * q1) / denom
            mean_re = corr_re.mean(dim=-1, keepdim=True)
            mean_im = corr_im.mean(dim=-1, keepdim=True)
            corr_features.append(torch.sqrt(mean_re.square() + mean_im.square() + self.eps))

        i_centered = i - i.mean(dim=-1, keepdim=True)
        q_centered = q - q.mean(dim=-1, keepdim=True)
        var_i = i_centered.square().mean(dim=-1, keepdim=True)
        var_q = q_centered.square().mean(dim=-1, keepdim=True)
        cov_iq = (i_centered * q_centered).mean(dim=-1, keepdim=True)
        trace = var_i + var_q
        anisotropy = torch.sqrt((var_i - var_q).square() + 4.0 * cov_iq.square() + self.eps) / (trace + self.eps)
        iq_corr = cov_iq / torch.sqrt(var_i * var_q + self.eps)
        iq_balance = (var_i - var_q) / (trace + self.eps)

        stats = [
            torch.log1p(energy.clamp_min(0.0)),
            torch.log1p(amp_var.clamp_min(0.0)),
            *corr_features,
            torch.log1p(trace.clamp_min(0.0)),
            anisotropy,
            iq_corr,
            iq_balance,
        ]
        return torch.cat(stats, dim=-1)


class RFILCMLocalMixer(nn.Module):
    """Multi-scale local branch used to isolate RF-statistical modulation."""

    def __init__(self, dim, kernel_sizes=(3, 7, 15)):
        super().__init__()
        self.local_convs = nn.ModuleList([
            nn.Conv1d(dim, dim, k, padding=k // 2, groups=dim, bias=False)
            for k in kernel_sizes
        ])
        self.scale_weight = nn.Parameter(torch.zeros(len(kernel_sizes)))

    def forward(self, x):
        xt = x.transpose(1, 2)
        weights = torch.softmax(self.scale_weight, dim=0)
        local = sum(w * conv(xt) for w, conv in zip(weights, self.local_convs))
        return local.transpose(1, 2)


def build_rf_stat_mask(stat_dim, mode, n_corr):
    """Create feature masks for RF-ILCM ablation variants."""
    if mode == "full":
        return None
    mask = torch.zeros(stat_dim, dtype=torch.float32)
    corr_start = 2
    corr_end = corr_start + n_corr
    iq_start = corr_end
    if mode == "corr":
        mask[corr_start:corr_end] = 1.0
    elif mode == "iq":
        mask[iq_start:] = 1.0
    else:
        raise ValueError(f"Unsupported RF-stat mask mode: {mode}")
    return mask


class RFILCMMixer(nn.Module):
    """
    RF-ILCM: RF-aware invariant local-correlation mixer.

    The mixer keeps local convolution as the main discriminative path and uses
    patch-level normalized-correlation / I/Q-shape statistics to modulate the
    local response. RF statistics are not added as standalone token features,
    which keeps the embedding space dominated by learned local temporal mixing.
    """
    uses_rf_stats = True

    def __init__(
        self,
        dim,
        stat_dim,
        kernel_sizes=(3, 7, 15),
        drop=0.0,
        stat_mask=None,
    ):
        super().__init__()
        self.local_convs = nn.ModuleList([
            nn.Conv1d(dim, dim, k, padding=k // 2, groups=dim, bias=False)
            for k in kernel_sizes
        ])
        self.scale_weight = nn.Parameter(torch.zeros(len(kernel_sizes)))

        stat_hidden = max(dim, stat_dim * 4)
        self.stat_encoder = nn.Sequential(
            nn.LayerNorm(stat_dim),
            nn.Linear(stat_dim, stat_hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(stat_hidden, dim),
            nn.GELU(),
        )
        self.channel_gamma = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.Tanh(),
        )
        self.token_gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, 1),
            nn.Sigmoid(),
        )
        self.post_proj = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(dim, dim),
        )
        self.gamma_scale = nn.Parameter(torch.tensor(0.5))
        self.res_scale = nn.Parameter(torch.tensor(1.0))
        if stat_mask is None:
            self.register_buffer("stat_mask", None)
        else:
            self.register_buffer("stat_mask", stat_mask.view(1, 1, -1))

    def forward(self, x, rf_stats=None):
        xt = x.transpose(1, 2)
        weights = torch.softmax(self.scale_weight, dim=0)
        local = sum(w * conv(xt) for w, conv in zip(weights, self.local_convs)).transpose(1, 2)

        if rf_stats is None:
            return local

        if self.stat_mask is not None:
            rf_stats = rf_stats * self.stat_mask

        stat = self.stat_encoder(rf_stats)
        gamma = self.channel_gamma(stat)
        token_gate = self.token_gate(torch.cat([x, stat], dim=-1))
        modulated = local * (1.0 + self.gamma_scale * gamma) * (0.5 + token_gate)
        return self.res_scale * self.post_proj(modulated)


class ConvAnchoredRFILCMMixer(nn.Module):
    """
    Conv-anchored RF-ILCM.

    A plain depthwise Conv path is kept as the stable fallback. The RF-aware
    multi-scale path only contributes through a learnable residual interpolation,
    so shortcut-sensitive RF statistics cannot fully replace the Conv response
    early in training.
    """
    uses_rf_stats = True

    def __init__(
        self,
        dim,
        stat_dim,
        conv_kernel_size=7,
        kernel_sizes=(3, 7, 15),
        drop=0.0,
        init_eta=0.1,
        stat_mask=None,
    ):
        super().__init__()
        self.conv_anchor = nn.Conv1d(
            dim, dim, conv_kernel_size,
            padding=conv_kernel_size // 2, groups=dim, bias=False,
        )
        self.rf_convs = nn.ModuleList([
            nn.Conv1d(dim, dim, k, padding=k // 2, groups=dim, bias=False)
            for k in kernel_sizes
        ])
        self.scale_weight = nn.Parameter(torch.zeros(len(kernel_sizes)))

        stat_hidden = max(dim, stat_dim * 4)
        self.stat_encoder = nn.Sequential(
            nn.LayerNorm(stat_dim),
            nn.Linear(stat_dim, stat_hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(stat_hidden, dim),
            nn.GELU(),
        )
        self.channel_gamma = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.Tanh(),
        )
        self.token_gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, 1),
            nn.Sigmoid(),
        )
        self.gamma_scale = nn.Parameter(torch.tensor(0.25))

        init_eta = float(min(max(init_eta, 1e-4), 1.0 - 1e-4))
        self.eta_logit = nn.Parameter(torch.tensor(math.log(init_eta / (1.0 - init_eta))))

        if stat_mask is None:
            self.register_buffer("stat_mask", None)
        else:
            self.register_buffer("stat_mask", stat_mask.view(1, 1, -1))

    def forward(self, x, rf_stats=None):
        xt = x.transpose(1, 2)
        conv = self.conv_anchor(xt).transpose(1, 2)

        if rf_stats is None:
            return conv

        weights = torch.softmax(self.scale_weight, dim=0)
        rf_local = sum(w * conv_layer(xt) for w, conv_layer in zip(weights, self.rf_convs))
        rf_local = rf_local.transpose(1, 2)

        if self.stat_mask is not None:
            rf_stats = rf_stats * self.stat_mask

        stat = self.stat_encoder(rf_stats)
        gamma = self.channel_gamma(stat)
        token_gate = self.token_gate(torch.cat([x, stat], dim=-1))
        rf_response = rf_local * (1.0 + self.gamma_scale * gamma) * (0.5 + token_gate)

        eta = torch.sigmoid(self.eta_logit)
        return conv + eta * (rf_response - conv)


class HybridMixer(nn.Module):
    """
    低层用ConvMixer（局部），高层用SpectralMixer（全局）
    通过stage参数控制当前是哪一层
    """
    def __init__(self, dim, num_tokens, stage, total_stages, kernel_size=7):
        super().__init__()
        if stage < total_stages // 2:
            self.mixer = ConvMixer(dim, kernel_size)
        else:
            self.mixer = SpectralMixer(num_tokens)

    def forward(self, x):
        return self.mixer(x)


# ─────────────────────────────────────────────
# MetaFormer Block
# ─────────────────────────────────────────────

class ChannelMLP(nn.Module):
    """MetaFormer中的Channel MLP（FFN）"""
    def __init__(self, dim, mlp_ratio=4, drop=0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, dim),
            nn.Dropout(drop),
        )

    def forward(self, x):
        return self.net(x)


class MetaFormerBlock(nn.Module):
    """
    标准MetaFormer Block:
        x = x + Mixer(Norm(x))
        x = x + MLP(Norm(x))
    """
    def __init__(self, dim, mixer, mlp_ratio=4, drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.mixer = mixer
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = ChannelMLP(dim, mlp_ratio, drop)

    def forward(self, x, rf_stats=None):
        normed = self.norm1(x)
        if getattr(self.mixer, "uses_rf_stats", False):
            mixed = self.mixer(normed, rf_stats=rf_stats)
        else:
            mixed = self.mixer(normed)
        x = x + mixed
        x = x + self.mlp(self.norm2(x))
        return x


# ─────────────────────────────────────────────
# Patch Embedding
# ─────────────────────────────────────────────

class RFPatchEmbedding(nn.Module):
    """
    将 [B, 2, L] 的IQ信号切成patch token序列 [B, N, D]
    用1D卷积实现，patch_size控制每个token覆盖的时域长度
    """
    def __init__(self, in_channels=2, dim=64, patch_size=48, signal_length=4800):
        super().__init__()
        assert signal_length % patch_size == 0, \
            f"signal_length {signal_length} must be divisible by patch_size {patch_size}"
        self.num_tokens = signal_length // patch_size
        self.proj = nn.Conv1d(in_channels, dim, kernel_size=patch_size, stride=patch_size, bias=False)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # x: [B, 2, L]
        x = self.proj(x)               # [B, D, N]
        x = x.transpose(1, 2)         # [B, N, D]
        return self.norm(x)


# ─────────────────────────────────────────────
# RF-MetaFormer
# ─────────────────────────────────────────────

MIXER_TYPES = [
    "identity",
    "pooling",
    "conv",
    "spectral",
    "hybrid",
    "attention",
    "mamba",
    "rfpmls",
    "rf_ilcm_local",
    "rf_ilcm_corr",
    "rf_ilcm_iq",
    "rf_ilcm",
    "rf_ilcm_anchor",
    "rf_ilcm_anchor_corr",
    "rf_ilcm_anchor_iq",
]


class RFMetaFormer(nn.Module):
    """
    RF-MetaFormer主干网络
    输入: [B, 2, L] IQ信号
    输出: [B, dim] 特征向量（用于few-shot分类）

    Args:
        mixer_type: 'identity' | 'pooling' | 'conv' | 'spectral' | 'hybrid' | 'attention'
        dim:        token特征维度
        depth:      MetaFormer Block层数
        patch_size: 每个token覆盖的时域点数
        signal_length: 输入信号长度
        mlp_ratio:  Channel MLP的扩展比
        drop:       Dropout率
    """

    def __init__(
        self,
        mixer_type="conv",
        dim=64,
        depth=4,
        patch_size=48,
        signal_length=4800,
        mlp_ratio=4,
        drop=0.0,
        conv_kernel_size=7,
        rf_anchor_init_eta=0.1,
    ):
        super().__init__()
        assert mixer_type in MIXER_TYPES, f"mixer_type must be one of {MIXER_TYPES}"

        self.mixer_type = mixer_type
        self.dim = dim
        self.conv_kernel_size = conv_kernel_size
        self.rf_anchor_init_eta = rf_anchor_init_eta

        self.patch_embed = RFPatchEmbedding(
            in_channels=2, dim=dim,
            patch_size=patch_size, signal_length=signal_length
        )
        self.rf_stats = RFLocalStatistics(patch_size=patch_size) if mixer_type in {
            "rf_ilcm_corr",
            "rf_ilcm_iq",
            "rf_ilcm",
            "rf_ilcm_anchor",
            "rf_ilcm_anchor_corr",
            "rf_ilcm_anchor_iq",
        } else None
        num_tokens = self.patch_embed.num_tokens

        self.blocks = nn.ModuleList([
            MetaFormerBlock(
                dim=dim,
                mixer=self._build_mixer(mixer_type, dim, num_tokens, stage=i, total_stages=depth),
                mlp_ratio=mlp_ratio,
                drop=drop,
            )
            for i in range(depth)
        ])

        self.norm = nn.LayerNorm(dim)

    def _build_mixer(self, mixer_type, dim, num_tokens, stage, total_stages):
        if mixer_type == "identity":
            return IdentityMixer()
        elif mixer_type == "pooling":
            return PoolingMixer(pool_size=3)
        elif mixer_type == "conv":
            return ConvMixer(dim, kernel_size=self.conv_kernel_size)
        elif mixer_type == "spectral":
            return SpectralMixer(num_tokens)
        elif mixer_type == "hybrid":
            return HybridMixer(dim, num_tokens, stage, total_stages, kernel_size=self.conv_kernel_size)
        elif mixer_type == "attention":
            return AttentionMixer(dim, num_heads=4)
        elif mixer_type == "mamba":
            return MambaMixer(dim)
        elif mixer_type == "rfpmls":
            return RFPMLSMixer(dim, num_tokens)
        elif mixer_type == "rf_ilcm_local":
            return RFILCMLocalMixer(dim)
        elif mixer_type in {
            "rf_ilcm_corr",
            "rf_ilcm_iq",
            "rf_ilcm",
            "rf_ilcm_anchor",
            "rf_ilcm_anchor_corr",
            "rf_ilcm_anchor_iq",
        }:
            stat_dim = self.rf_stats.stat_dim if self.rf_stats is not None else RFLocalStatistics(48).stat_dim
            if self.rf_stats is not None:
                n_corr = len(self.rf_stats.lags)
            else:
                n_corr = len(RFLocalStatistics(48).lags)
            mask_mode = {
                "rf_ilcm_corr": "corr",
                "rf_ilcm_iq": "iq",
                "rf_ilcm": "full",
                "rf_ilcm_anchor": "full",
                "rf_ilcm_anchor_corr": "corr",
                "rf_ilcm_anchor_iq": "iq",
            }[mixer_type]
            stat_mask = build_rf_stat_mask(stat_dim, mask_mode, n_corr)
            if mixer_type in {"rf_ilcm_anchor", "rf_ilcm_anchor_corr", "rf_ilcm_anchor_iq"}:
                return ConvAnchoredRFILCMMixer(
                    dim,
                    stat_dim,
                    conv_kernel_size=self.conv_kernel_size,
                    drop=0.0,
                    init_eta=self.rf_anchor_init_eta,
                    stat_mask=stat_mask,
                )
            return RFILCMMixer(dim, stat_dim, drop=0.0, stat_mask=stat_mask)

    def forward(self, x):
        # x: [B, 2, L]
        rf_stats = self.rf_stats(x) if self.rf_stats is not None else None
        x = self.patch_embed(x)        # [B, N, D]
        for block in self.blocks:
            x = block(x, rf_stats=rf_stats)               # [B, N, D]
        x = self.norm(x)
        x = x.mean(dim=1)             # [B, D] 全局平均池化
        return x

    def extra_repr(self):
        return f"mixer_type={self.mixer_type}, dim={self.dim}, conv_kernel_size={self.conv_kernel_size}"
