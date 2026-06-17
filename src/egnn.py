import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from src import utils
from src.condition_model import GatedFiLMGuidanceModule

class RBFEmbedding(nn.Module):
    def __init__(self, start=0.0, stop=5.0, num_gaussians=16):
        super().__init__()
        # 定义一系列高斯核的中心点
        offset = torch.linspace(start, stop, num_gaussians)
        self.register_buffer('offset', offset)
        
        # 定义高斯核的宽度 (根据中心点间距自动计算)
        width = offset[1] - offset[0]
        self.coeff = -0.5 / (width ** 2)

    def forward(self, dist):
        # dist: [N, 1]
        # output: [N, num_gaussians]
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))

def timestep_embedding(t, dim, max_period=10000, time_factor: float = 1000.0):
    """
    Create sinusoidal timestep embeddings.
    :param t: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an (N, D) Tensor of positional embeddings.
    """
    t = time_factor * t
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
        t.device
    )

    args = t.float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    if torch.is_floating_point(t):
        embedding = embedding.to(t)
    return embedding

class GCL(nn.Module):
    def __init__(self, input_nf, output_nf, hidden_nf, normalization_factor, aggregation_method, activation,
                 edges_in_d=0, nodes_att_dim=0, attention=False, normalization=None):
        super(GCL, self).__init__()
        input_edge = input_nf * 2
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method
        self.attention = attention

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edges_in_d, hidden_nf),
            activation,
            nn.Linear(hidden_nf, hidden_nf),
            activation)

        if normalization is None:
            self.node_mlp = nn.Sequential(
                nn.Linear(hidden_nf + input_nf + nodes_att_dim, hidden_nf),
                activation,
                nn.Linear(hidden_nf, output_nf)
            )
        elif normalization == 'batch_norm':
            self.node_mlp = nn.Sequential(
                nn.Linear(hidden_nf + input_nf + nodes_att_dim, hidden_nf),
                nn.BatchNorm1d(hidden_nf),
                activation,
                nn.Linear(hidden_nf, output_nf),
                nn.BatchNorm1d(output_nf),
            )
        else:
            raise NotImplementedError

        if self.attention:
            self.att_mlp = nn.Sequential(nn.Linear(hidden_nf, 1), nn.Sigmoid())

    def edge_model(self, source, target, edge_attr, edge_mask, edge_gate):
        if edge_attr is None:  # Unused.
            out = torch.cat([source, target], dim=1)
        else:
            out = torch.cat([source, target, edge_attr], dim=1)
        mij = self.edge_mlp(out)

        if self.attention:
            att_val = self.att_mlp(mij)
            out = mij * att_val
        else:
            out = mij
        
        if edge_gate is not None:
            out = out * edge_gate

        if edge_mask is not None:
            out = out * edge_mask
        return out, mij

    def node_model(self, x, edge_index, edge_attr, node_attr):
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0),
                                   normalization_factor=self.normalization_factor,
                                   aggregation_method=self.aggregation_method)
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        out = x + self.node_mlp(agg)
        return out, agg

    def forward(self, h, edge_index, edge_attr=None, node_attr=None, node_mask=None, edge_mask=None, edge_gate=None):
        row, col = edge_index
        edge_feat, mij = self.edge_model(h[row], h[col], edge_attr, edge_mask, edge_gate)
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)
        if node_mask is not None:
            h = h * node_mask
        return h, mij


class EquivariantUpdate(nn.Module):
    def __init__(self, hidden_nf, normalization_factor, aggregation_method,
                 edges_in_d=1, activation=nn.SiLU(), tanh=False, coords_range=10.0):
        super(EquivariantUpdate, self).__init__()
        self.tanh = tanh
        self.coords_range = coords_range
        input_edge = hidden_nf * 2 + edges_in_d
        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)
        self.coord_mlp = nn.Sequential(
            nn.Linear(input_edge, hidden_nf),
            activation,
            nn.Linear(hidden_nf, hidden_nf),
            activation,
            layer)
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method

    def coord_model(self, h, coord, edge_index, coord_diff, edge_attr, edge_mask, rgroup_mask, edge_gate):
        row, col = edge_index
        input_tensor = torch.cat([h[row], h[col], edge_attr], dim=1)
        if self.tanh:
            trans = coord_diff * torch.tanh(self.coord_mlp(input_tensor)) * self.coords_range
        else:
            trans = coord_diff * self.coord_mlp(input_tensor)
        if edge_gate is not None:
            trans = trans * edge_gate
        if edge_mask is not None:
            trans = trans * edge_mask
        agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0),
                                   normalization_factor=self.normalization_factor,
                                   aggregation_method=self.aggregation_method)
        if rgroup_mask is not None:
            agg = agg * rgroup_mask
        coord = coord + agg
        return coord

    def forward(
            self, h, coord, edge_index, coord_diff, edge_attr=None, rgroup_mask=None, node_mask=None, edge_mask=None, edge_gate=None
    ):
        coord = self.coord_model(h, coord, edge_index, coord_diff, edge_attr, edge_mask, rgroup_mask, edge_gate)
        if node_mask is not None:
            coord = coord * node_mask
        return coord

class EquivariantBlock(nn.Module):
    def __init__(self, hidden_nf, edge_feat_nf=2, device='cpu', activation=nn.SiLU(), n_layers=2, attention=True,
                 norm_diff=True, tanh=False, coords_range=15, norm_constant=1, sin_embedding=None,
                 normalization_factor=100, aggregation_method='sum'):
        super(EquivariantBlock, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.coords_range_layer = float(coords_range)
        self.norm_diff = norm_diff
        self.norm_constant = norm_constant
        self.sin_embedding = sin_embedding
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method

        self.RBFEmbedding = RBFEmbedding(start=0.0, stop=5.0, num_gaussians=hidden_nf)

        self.atom_gate_mlp = nn.Sequential(
            nn.Linear(2 * hidden_nf, 64),
            activation,
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=edge_feat_nf,
                                              activation=activation, attention=attention,
                                              normalization_factor=self.normalization_factor,
                                              aggregation_method=self.aggregation_method))
        self.add_module("gcl_equiv", EquivariantUpdate(hidden_nf, edges_in_d=edge_feat_nf, activation=activation, tanh=tanh,
                                                       coords_range=self.coords_range_layer,
                                                       normalization_factor=self.normalization_factor,
                                                       aggregation_method=self.aggregation_method))
        self.to(self.device)

    def forward(self, h, x, edge_index, node_mask=None, rgroup_mask=None, edge_mask=None, edge_attr=None):
        # Edit Emiel: Remove velocity as input
        distances, coord_diff = coord2diff(x, edge_index, self.norm_constant)
        if self.sin_embedding is not None:
            distances = self.sin_embedding(distances)
        edge_attr = torch.cat([distances, edge_attr], dim=1)

        # Compute distance to anchor(0,0,0)
        dist_to_anchor = torch.norm(x, dim=-1, keepdim=True)
        dist_anchor_emb = self.RBFEmbedding(dist_to_anchor)
        gate_input = torch.cat([h, dist_anchor_emb], dim=-1)
        gate = (self.atom_gate_mlp(gate_input) * rgroup_mask + (1 - rgroup_mask)) * node_mask

        # Compute edge_gate
        row, col = edge_index
        p_real_receiver = gate[row]
        p_fake_sender = 1 - gate[col]
        edge_gate = 1 - p_real_receiver * p_fake_sender

        for i in range(0, self.n_layers):
            h, _ = self._modules["gcl_%d" % i](h, edge_index, edge_attr=edge_attr, node_mask=node_mask, edge_mask=edge_mask, edge_gate = edge_gate)
        x = self._modules["gcl_equiv"](
            h, x,
            edge_index=edge_index,
            coord_diff=coord_diff,
            edge_attr=edge_attr,
            rgroup_mask=rgroup_mask,
            node_mask=node_mask,
            edge_mask=edge_mask,
            edge_gate = edge_gate
        )

        # Important, the bias of the last linear might be non-zero
        if node_mask is not None:
            h = h * node_mask
        return h, x


class EGNN(nn.Module):
    def __init__(self, hidden_nf, device='cpu', activation=nn.SiLU(), n_layers=3, attention=False,
                 norm_diff=True, out_node_nf=None, tanh=False, coords_range=15, norm_constant=1, inv_sublayers=2,
                 sin_embedding=False, normalization_factor=100, aggregation_method='sum'):
        super(EGNN, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.coords_range_layer = float(coords_range/n_layers)
        self.norm_diff = norm_diff
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method
        

        if sin_embedding:
            self.sin_embedding = SinusoidsEmbeddingNew()
            edge_feat_nf = self.sin_embedding.dim * 2
        else:
            self.sin_embedding = None
            edge_feat_nf = 2
        self.embedding_out = nn.Linear(self.hidden_nf, out_node_nf)
        for i in range(0, n_layers):
            self.add_module("e_block_%d" % i, EquivariantBlock(hidden_nf, edge_feat_nf=edge_feat_nf, device=device,
                                                               activation=activation, n_layers=inv_sublayers,
                                                               attention=attention, norm_diff=norm_diff, tanh=tanh,
                                                               coords_range=coords_range, norm_constant=norm_constant,
                                                               sin_embedding=self.sin_embedding,
                                                               normalization_factor=self.normalization_factor,
                                                               aggregation_method=self.aggregation_method))
        for i in range(0, n_layers):
            self.add_module("film_block%d" % i, GatedFiLMGuidanceModule(hidden_nf))

        self.to(self.device)

    def forward(self, h, x, bs, n_nodes, edge_index, node_mask=None, rgroup_mask=None, edge_mask=None, time_embedding = None, affinity_condition = None):
        # Edit Emiel: Remove velocity as input
        distances, _ = coord2diff(x, edge_index)
        if self.sin_embedding is not None:
            distances = self.sin_embedding(distances)

        h_current = h
        for i in range(0, self.n_layers):
            h_film = self._modules["film_block%d" % i](h=h_current, t=time_embedding, c=affinity_condition,node_mask = node_mask, rgroup_mask = rgroup_mask)
            h_current, x = self._modules["e_block_%d" % i](
                h_film, x, edge_index,
                node_mask=node_mask,
                rgroup_mask=rgroup_mask,
                edge_mask=edge_mask,
                edge_attr=distances
            )#这里h其实都更新了，但是x并没有更新pocket和scaffold部分
        
        # Important, the bias of the last linear might be non-zero
        h = self.embedding_out(h_current)
        if node_mask is not None:
            h = h * node_mask #这个地方消除padding的影响
        return h, x


class GNN(nn.Module):
    def __init__(self, in_node_nf, in_edge_nf, hidden_nf, aggregation_method='sum', device='cpu',
                 activation=nn.SiLU(), n_layers=4, attention=False, normalization_factor=1,
                 out_node_nf=None, normalization=None):
        super(GNN, self).__init__()
        if out_node_nf is None:
            out_node_nf = in_node_nf
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers

        # Encoder
        self.embedding = nn.Linear(in_node_nf, self.hidden_nf)
        self.embedding_out = nn.Linear(self.hidden_nf, out_node_nf)
        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, GCL(
                self.hidden_nf, self.hidden_nf, self.hidden_nf,
                normalization_factor=normalization_factor,
                aggregation_method=aggregation_method,
                edges_in_d=in_edge_nf, activation=activation,
                attention=attention, normalization=normalization))
        self.to(self.device)

    def forward(self, h, edges, edge_attr=None, node_mask=None, edge_mask=None):
        # Edit Emiel: Remove velocity as input
        h = self.embedding(h)
        for i in range(0, self.n_layers):
            h, _ = self._modules["gcl_%d" % i](h, edges, edge_attr=edge_attr, node_mask=node_mask, edge_mask=edge_mask)
        h = self.embedding_out(h)

        # Important, the bias of the last linear might be non-zero
        if node_mask is not None:
            h = h * node_mask
        return h


class SinusoidsEmbeddingNew(nn.Module):
    def __init__(self, max_res=15., min_res=15. / 2000., div_factor=4):
        super().__init__()
        self.n_frequencies = int(math.log(max_res / min_res, div_factor)) + 1
        self.frequencies = 2 * math.pi * div_factor ** torch.arange(self.n_frequencies)/max_res
        self.dim = len(self.frequencies) * 2

    def forward(self, x):
        x = torch.sqrt(x + 1e-8)
        emb = x * self.frequencies[None, :].to(x.device)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb.detach()


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


class Dynamics(nn.Module):
    def __init__(
            self, n_dims, in_node_nf, context_node_nf, hidden_nf=64, device='cpu', activation=nn.SiLU(),
            n_layers=4, attention=False, condition_time=True, tanh=False, norm_constant=0, inv_sublayers=2,
            sin_embedding=False, normalization_factor=100, aggregation_method='sum', model='egnn_dynamics',
            normalization=None, centering=False,
    ):
        super().__init__()
        self.device = device
        self.n_dims = n_dims
        self.context_node_nf = context_node_nf
        self.condition_time = condition_time
        self.model = model
        self.centering = centering
        self.hidden_nf = hidden_nf
        self.time_embedding_dim = hidden_nf // 2
        self.origin_node_nf = in_node_nf
        

        out_node_nf = self.origin_node_nf
        if self.model == 'egnn_dynamics':
            self.dynamics = EGNN(
                out_node_nf=out_node_nf,
                hidden_nf=hidden_nf, device=device,
                activation=activation,
                n_layers=n_layers,
                attention=attention,
                tanh=tanh,
                norm_constant=norm_constant,
                inv_sublayers=inv_sublayers,
                sin_embedding=sin_embedding,
                normalization_factor=normalization_factor,
                aggregation_method=aggregation_method,
            )
        elif self.model == 'gnn_dynamics':
            self.dynamics = GNN(
                in_node_nf=in_node_nf+3,
                in_edge_nf=0,
                hidden_nf=hidden_nf,
                out_node_nf=in_node_nf+3,
                device=device,
                activation=activation,
                n_layers=n_layers,
                attention=attention,
                normalization_factor=normalization_factor,
                aggregation_method=aggregation_method,
                normalization=normalization,
            )
        else:
            raise NotImplementedError
        
        self.atom_embedding = nn.Linear(self.origin_node_nf + context_node_nf, self.hidden_nf)
        self.time_mlp = nn.Sequential(
            nn.Linear(self.time_embedding_dim, self.hidden_nf),
            nn.SiLU(),
            nn.Linear(self.hidden_nf, self.hidden_nf)
        )
        self.edge_cache = {}

    def forward(self, t, xh, node_mask, rgroup_mask, edge_mask, context):
        """
        - t: (B)
        - xh: (B, N, D), where D = 3 + nf
        - node_mask: (B, N, 1)
        - edge_mask: (B*N, 1)
        - context: (B, N, C)
        """

        bs, n_nodes = xh.shape[0], xh.shape[1]
        edges = self.get_edges(n_nodes, bs)  # (2, B*N)
        node_mask = node_mask.view(bs * n_nodes, 1)  # (B*N, 1)

        if rgroup_mask is not None:
            rgroup_mask = rgroup_mask.view(bs * n_nodes, 1)  # (B*N, 1)

        # Reshaping node features & adding time feature
        xh = xh.view(bs * n_nodes, -1).clone() * node_mask  # (B*N, D) #这一步是将padding的部分影响消除
        x = xh[:, :self.n_dims].clone()  # (B*N, 3)
        h = xh[:, self.n_dims:].clone()  # (B*N, nf)
        if self.condition_time:
            if np.prod(t.size()) == 1:
                # t is the same for all elements in batch.
                h_time = torch.empty_like(h[:, 0:1]).fill_(t.item())
            else:
                # t is different over the batch dimension.
                h_time = t.view(bs, 1).repeat(1, n_nodes)
                h_time = h_time.view(bs * n_nodes, 1)
            # Emdding time step t
            h_time = timestep_embedding(h_time, self.time_embedding_dim) #(B*N, nt)
            h = torch.cat([h, h_time], dim=1)  # (B*N, nf+nt)
        if context is not None:
            context = context.view(bs*n_nodes, self.context_node_nf)
            h = torch.cat([h, context], dim=1)

        # Forward EGNN
        # Output: h_final (B*N, nf), x_final (B*N, 3), vel (B*N, 3)
        if self.model == 'egnn_dynamics':
            h_final, x_final = self.dynamics(
                h,
                x,
                bs,
                n_nodes,
                edges,
                node_mask=node_mask,
                rgroup_mask=rgroup_mask,
                edge_mask=edge_mask,
            )
            vel = (x_final - x) * node_mask  # This masking operation is redundant but just in case
        elif self.model == 'gnn_dynamics':
            xh = torch.cat([x, h], dim=1)
            output = self.dynamics(xh, edges, node_mask=node_mask)
            vel = output[:, 0:3] * node_mask
            h_final = output[:, 3:]
        else:
            raise NotImplementedError

        # Slice off context size
        if context is not None:
            h_final = h_final[:, :-self.context_node_nf]

        # Slice off last dimension which represented time.
        if self.condition_time:
            h_final = h_final[:, :-1]

        vel = vel.view(bs, n_nodes, -1)  # (B, N, 3)
        h_final = h_final.view(bs, n_nodes, -1)  # (B, N, D)
        node_mask = node_mask.view(bs, n_nodes, 1)  # (B, N, 1)

        if torch.any(torch.isnan(vel)) or torch.any(torch.isnan(h_final)):
            raise utils.FoundNaNException(vel, h_final)

        if self.centering:
            vel = utils.remove_mean_with_mask(vel, node_mask)

        return torch.cat([vel, h_final], dim=2)

    def get_edges(self, n_nodes, batch_size):
        if n_nodes in self.edge_cache:
            edges_dic_b = self.edge_cache[n_nodes]
            if batch_size in edges_dic_b:
                return edges_dic_b[batch_size]
            else:
                # get edges for a single sample
                rows, cols = [], []
                for batch_idx in range(batch_size):
                    for i in range(n_nodes):
                        for j in range(n_nodes):
                            rows.append(i + batch_idx * n_nodes)
                            cols.append(j + batch_idx * n_nodes)
                edges = [torch.LongTensor(rows).to(self.device), torch.LongTensor(cols).to(self.device)]
                edges_dic_b[batch_size] = edges
                return edges
        else:
            self.edge_cache[n_nodes] = {}
            return self.get_edges(n_nodes, batch_size)

    
class SimpleDynamics(Dynamics):
    def __init__(
            self,n_dims, in_node_nf, context_node_nf, hidden_nf=64, device='cpu', activation=nn.SiLU(),
            n_layers=4, attention=False, condition_time=True, tanh=False, norm_constant=0, inv_sublayers=2,
            sin_embedding=False, normalization_factor=100, aggregation_method='sum', model='egnn_dynamics',
            normalization=None, centering=False,
    ):  
        super().__init__(n_dims, in_node_nf, context_node_nf, hidden_nf, device, activation,
                         n_layers, attention, condition_time, tanh, norm_constant, inv_sublayers,
                         sin_embedding, normalization_factor, aggregation_method, model, normalization,
                         centering)
        self.hidden_nf = hidden_nf
        self.null_affinity_emb = nn.Parameter(torch.randn(1, self.hidden_nf))
        self.affinity_sin_embedding = SinusoidalEmbedding(dim = hidden_nf // 2)
        self.affinity_mlp = nn.Sequential(
            nn.Linear(hidden_nf // 2, hidden_nf),
            nn.SiLU(),
            nn.Linear(hidden_nf, hidden_nf)
        )

    #edge_mask实际没用到，因为edges是按照distance来的，已经重构
    def forward(self, t, xh, node_mask, rgroup_mask, context, affinity_condition, batch_mask, p_uncond = 0.1): 
        """
        - t: (B)
        - xh: (B, N, D), where D = 3 + nf
        - node_mask: (B, N, 1)
        - batch_mask: (B*N, 1)
        - context: (B, N, C)
        """
        #tranfer device of null_affinity_emb to support DDP
        self.null_affinity_emb.to(xh.device)

        bs, n_nodes = xh.shape[0], xh.shape[1]
        node_mask = node_mask.view(bs * n_nodes, 1)  # (B*N, 1)

        if rgroup_mask is not None:
            rgroup_mask = rgroup_mask.view(bs * n_nodes, 1)  # (B*N, 1)

        # Reshaping node features & adding time feature
        xh = xh.view(bs * n_nodes, -1).clone() * node_mask  # (B*N, D)
        x = xh[:, :self.n_dims].clone()  # (B*N, 3)
        h = xh[:, self.n_dims:].clone()  # (B*N, nf)

        # 依据4A和batch_index创建边
        edges_egnn1 = self.get_dist_edges(x, node_mask, batch_mask)

        if context is not None:
            context = context.view(bs*n_nodes, self.context_node_nf)
            h = torch.cat([h, context], dim=1)
            h_atom = self.atom_embedding(h)

        h_time = None
        if self.condition_time:
            if np.prod(t.size()) == 1:
                # t is the same for all elements in batch.
                h_time = torch.empty_like(h[:, 0:1]).fill_(t.item())
            else:
                # t is different over the batch dimension.
                h_time = t.view(bs, 1).repeat(1, n_nodes)
                h_time = h_time.view(bs * n_nodes, 1)
            # Emdding time step t
            h_time_sin_embedded = timestep_embedding(h_time, self.time_embedding_dim) #(B*N, nt)
            h_time = self.time_mlp(h_time_sin_embedded)

        # Build affinity condition
        affinity_condition = self.affinity_sin_embedding(affinity_condition)
        affinity_condition = self.affinity_mlp(affinity_condition)
        # Replace affinity condition by null condition with certrain probability
        final_affinity_condition = self.cfg_condition_dropout(affinity_condition, p_uncond)
        final_affinity_condition = torch.repeat_interleave(final_affinity_condition, repeats=n_nodes, dim=0)

        # Forward EGNN
        # Output: h_final (B*N, nf), x_final (B*N, 3), vel (B*N, 3)
        h_final, x_final = self.dynamics(
            h_atom,
            x,
            bs,
            n_nodes,
            edges_egnn1,
            node_mask=node_mask,
            rgroup_mask=rgroup_mask,
            edge_mask=None, 
            time_embedding = h_time,
            affinity_condition = final_affinity_condition        
        )

        vel_refined = (x_final - x) * node_mask

        vel_refined = vel_refined.view(bs, n_nodes, -1)  # (B, N, 3)
        h_final = h_final.view(bs, n_nodes, -1)  # (B, N, D)
        node_mask = node_mask.view(bs, n_nodes, 1)  # (B, N, 1)

        if torch.any(torch.isnan(vel_refined)) or torch.any(torch.isnan(h_final)):
            raise utils.FoundNaNException(vel_refined, h_final)

        if self.centering:
            vel_refined = utils.remove_mean_with_mask(vel_refined, node_mask)

        return torch.cat([vel_refined, h_final], dim=2)
    
    def cfg_condition_dropout(self, affinity_embedded, p_uncond):
        if p_uncond < 0.0 or p_uncond > 1.0:
            raise ValueError("p_uncond must be between 0.0 and 1.0")

        if p_uncond == 0.0: # Optimization: If no dropout, return original
            return affinity_embedded
        
        if p_uncond == 1.0: # This condition used when sample need no condition generation
            return self.null_affinity_emb.expand_as(affinity_embedded)
        
        batch_size = affinity_embedded.shape[0]
        dropout_mask = (torch.rand(batch_size, 1, device = affinity_embedded.device) < p_uncond)

        final_affinity_embedded = torch.where(
            dropout_mask,
            self.null_affinity_emb,
            affinity_embedded
        )
        return final_affinity_embedded


    @staticmethod
    def get_dist_edges(x, node_mask, batch_mask):
        node_mask = node_mask.squeeze().bool()
        batch_adj = (batch_mask[:, None] == batch_mask[None, :])
        nodes_adj = (node_mask[:, None] & node_mask[None, :])
        dists_adj = (torch.cdist(x, x) <= 4)
        rm_self_loops = ~torch.eye(x.size(0), dtype=torch.bool, device=x.device)
        adj = batch_adj & nodes_adj & dists_adj & rm_self_loops
        edges = torch.stack(torch.where(adj))
        return edges
    
class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        # x 形状: (B, 1)
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

