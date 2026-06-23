import torch.nn as nn
import torch
import math
import torch.nn.functional as F

def coord2diff(x, edge_index, norm_constant=1):
    row, col = edge_index
    coord_diff = x[row] - x[col]
    radial = torch.sum((coord_diff) ** 2, 1).unsqueeze(1)
    norm = torch.sqrt(radial + 1e-8)
    coord_diff = coord_diff/(norm + norm_constant)
    return radial, coord_diff

def unsorted_segment_sum(data, segment_ids, num_segments, normalization_factor, aggregation_method: str):
    """Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`.
        Normalization: 'sum' or 'mean'.
    """
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    if aggregation_method == 'sum':
        result = result / normalization_factor

    if aggregation_method == 'mean':
        norm = data.new_zeros(result.shape)
        norm.scatter_add_(0, segment_ids, data.new_ones(data.shape))
        norm[norm == 0] = 1
        result = result / norm
    return result

class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        # x shape: (B, 1)
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class GatedFiLMGuidanceModule(nn.Module):
    """
    Residual gated FiLM module for affinity-guided EGNN.
    """

    def __init__(
        self,
        hidden_dim: int,
        init_rgroup_scale: float = 1.0,
        init_context_scale: float = 1.0,
        zero_init_film: bool = True,
        gate_init_bias: float = -2.0,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        self.layer_norm = nn.LayerNorm(hidden_dim)

        # Time branch: denoising-stage modulation.
        self.time_film_generator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim),
        )

        # Affinity branch: target-affinity modulation.
        self.affinity_film_generator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim),
        )

        # Node-wise gate: different nodes receive different conditioning strength.
        self.gate_net = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )

        # Positive learnable scales for affinity modulation.
        # softplus keeps them positive while allowing values > 1.
        self.rgroup_scale_logit = nn.Parameter(
            self._inverse_softplus(torch.tensor(float(init_rgroup_scale)))
        )
        self.context_scale_logit = nn.Parameter(
            self._inverse_softplus(torch.tensor(float(init_context_scale)))
        )

        if zero_init_film:
            # Start from near-identity:
            # film_update ~= 0, so h_out ~= h.
            nn.init.zeros_(self.time_film_generator[-1].weight)
            nn.init.zeros_(self.time_film_generator[-1].bias)
            nn.init.zeros_(self.affinity_film_generator[-1].weight)
            nn.init.zeros_(self.affinity_film_generator[-1].bias)

        # Conservative initial gate.
        # gate ~= sigmoid(-2) ~= 0.12.
        nn.init.zeros_(self.gate_net[-2].weight)
        nn.init.constant_(self.gate_net[-2].bias, gate_init_bias)

    @staticmethod
    def _inverse_softplus(x: torch.Tensor) -> torch.Tensor:
        """
        Inverse of softplus for positive initialization.
        """
        x = torch.clamp(x, min=1e-6)
        return torch.log(torch.expm1(x))

    def forward(
        self,
        h: torch.Tensor,
        t: torch.Tensor,
        c: torch.Tensor,
        node_mask: torch.Tensor = None,
        rgroup_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            h:
                [B*N, hidden_dim], current node hidden states.
            t:
                [B*N, hidden_dim], per-node time embedding.
            c:
                [B*N, hidden_dim], per-node affinity condition embedding.
            node_mask:
                [B*N, 1], 1 for valid nodes, 0 for padding nodes.
            rgroup_mask:
                [B*N, 1], 1 for R-chain/R-group nodes,
                0 for scaffold-pocket environment nodes.

        Returns:
            h_out:
                [B*N, hidden_dim], modulated node hidden states.
        """

        if t is None:
            raise ValueError("time embedding t cannot be None.")
        if c is None:
            raise ValueError("affinity condition c cannot be None.")

        h_norm = self.layer_norm(h)

        # 1. Decoupled FiLM parameters.
        delta_gamma_t, beta_t = self.time_film_generator(t).chunk(2, dim=-1)
        delta_gamma_c, beta_c = self.affinity_film_generator(c).chunk(2, dim=-1)

        # 2. Learnable affinity modulation strength for R-chain and context.
        if rgroup_mask is not None:
            rgroup_scale = F.softplus(self.rgroup_scale_logit)
            context_scale = F.softplus(self.context_scale_logit)

            type_scale = (
                rgroup_mask * rgroup_scale
                + (1.0 - rgroup_mask) * context_scale
            )
        else:
            type_scale = 1.0

        # Time modulation is shared by all nodes.
        # Affinity modulation is scaled by node role.
        delta_gamma = delta_gamma_t + type_scale * delta_gamma_c
        beta = beta_t + type_scale * beta_c

        film_update = delta_gamma * h_norm + beta

        # 3. Node-wise gate.
        gate_input = torch.cat([h_norm, t, c], dim=-1)
        gate = self.gate_net(gate_input)

        # 4. Residual FiLM.
        h_out = h + gate * film_update

        # 5. Keep padding nodes zero.
        if node_mask is not None:
            h_out = h_out * node_mask

        return h_out
