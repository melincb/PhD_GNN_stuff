import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, LayerNorm, DiffGroupNorm
from torch.utils.checkpoint import checkpoint
from typing import List, Dict

# class AttentionPooling(nn.Module):
#     """
#     Attention-based pooling that aggregates fine-scale node features 
#     to coarse-scale nodes using explicit hierarchical connections.
#     """
#     def __init__(self, feature_dim: int, heads: int = 4, dropout: float = 0.1):
#         super().__init__()
#         self.feature_dim = feature_dim
#         self.heads = heads
#         self.dropout = dropout
        
#         # Attention mechanism for pooling
#         self.query_proj = nn.Linear(feature_dim, feature_dim)
#         self.key_proj = nn.Linear(feature_dim, feature_dim)
#         self.value_proj = nn.Linear(feature_dim, feature_dim)
        
#         # Multi-head setup
#         self.head_dim = feature_dim // heads
#         assert feature_dim % heads == 0, "feature_dim must be divisible by heads"
        
#         self.scale = self.head_dim ** -0.5
#         self.dropout_layer = nn.Dropout(dropout)
        
#         # Optional projection after pooling
#         self.output_proj = nn.Linear(feature_dim, feature_dim)
        
#     def forward(self, fine_features: torch.Tensor, fine_to_coarse_connections: Dict[int, List[int]], coarse_idx_to_output_idx: Dict[int, int]) -> torch.Tensor:
#         """
#         Args:
#             fine_features: [N_fine, feature_dim] - features of fine-scale nodes
#             fine_to_coarse_connections: dict mapping coarse node indices to lists of fine node indices
#                                       e.g., {36: [36, 35, 37, 38, 34, 13, 69, 68], ...}
        
#         Returns:
#             coarse_features: [N_coarse, feature_dim] - aggregated features for coarse nodes
#         """
#         N_fine, feature_dim = fine_features.shape
#         N_coarse = len(fine_to_coarse_connections)
#         device = fine_features.device

#         # Project to query, key, value
#         Q = self.query_proj(fine_features)  # [N_fine, feature_dim]
#         K = self.key_proj(fine_features)    # [N_fine, feature_dim]
#         V = self.value_proj(fine_features)  # [N_fine, feature_dim]
        
#         # Reshape for multi-head attention
#         Q = Q.view(N_fine, self.heads, self.head_dim)  # [N_fine, heads, head_dim]
#         K = K.view(N_fine, self.heads, self.head_dim)  # [N_fine, heads, head_dim]
#         V = V.view(N_fine, self.heads, self.head_dim)  # [N_fine, heads, head_dim]
        
#         # Initialize coarse features
#         coarse_features = torch.zeros(N_coarse, self.heads, self.head_dim, 
#                                     device=device, dtype=fine_features.dtype)
        
#         # Create mapping from coarse index in connections to output index
#         # coarse_indices = sorted(fine_to_coarse_connections.keys())
#         # coarse_idx_to_output_idx = {coarse_idx: i for i, coarse_idx in enumerate(coarse_indices)}
        

#         # For each coarse node, attend to its connected fine nodes
#         for coarse_idx, connected_fine_indices in fine_to_coarse_connections.items():
#             output_idx = coarse_idx_to_output_idx[coarse_idx]
            
#             # Convert to tensor and filter valid indices
#             connected_fine_indices = torch.tensor(connected_fine_indices, device=device)
#             # Only keep indices that are within bounds
#             valid_mask = connected_fine_indices < N_fine
#             connected_fine_indices = connected_fine_indices[valid_mask]
            
#             if len(connected_fine_indices) == 0:
#                 continue
                
#             # Get features for connected fine nodes
#             q_connected = Q[connected_fine_indices]  # [n_connected, heads, head_dim]
#             k_connected = K[connected_fine_indices]  # [n_connected, heads, head_dim]
#             v_connected = V[connected_fine_indices]  # [n_connected, heads, head_dim]
            
#             # Compute attention scores within the connected group
#             scores = torch.einsum('nhd,mhd->nhm', q_connected, k_connected) * self.scale
#             attention_weights = F.softmax(scores, dim=-1)  # [n_connected, heads, n_connected]
#             attention_weights = self.dropout_layer(attention_weights)
            
#             # Apply attention to values and aggregate
#             attended_values = torch.einsum('nhm,mhd->nhd', attention_weights, v_connected)
#             # Pool by taking mean across the connected fine nodes
#             coarse_features[output_idx] = attended_values.mean(dim=0)

#         # Reshape back and apply output projection
#         coarse_features = coarse_features.view(N_coarse, feature_dim)
#         coarse_features = self.output_proj(coarse_features)
        
#         return coarse_features

# class AttentionUnpooling(nn.Module):
#     """
#     Attention-based unpooling that distributes coarse-scale features 
#     to fine-scale nodes using explicit hierarchical connections.
#     """
#     def __init__(self, feature_dim: int, heads: int = 4, dropout: float = 0.1):
#         super().__init__()
#         self.feature_dim = feature_dim
#         self.heads = heads
#         self.dropout = dropout
        
#         # Attention mechanism for unpooling
#         self.query_proj = nn.Linear(feature_dim, feature_dim)
#         self.key_proj = nn.Linear(feature_dim, feature_dim)
#         self.value_proj = nn.Linear(feature_dim, feature_dim)
        
#         # Multi-head setup
#         self.head_dim = feature_dim // heads
#         assert feature_dim % heads == 0, "feature_dim must be divisible by heads"
        
#         self.scale = self.head_dim ** -0.5
#         self.dropout_layer = nn.Dropout(dropout)
        
#         # Learnable interpolation weights
#         self.interpolation_proj = nn.Linear(feature_dim, feature_dim)
        
#     def forward(self, coarse_features: torch.Tensor, coarse_to_fine_connections: Dict[int, List[int]], 
#                 fine_graph_edges: torch.Tensor) -> torch.Tensor:
#         """
#         Args:
#             coarse_features: [N_coarse, feature_dim] - features of coarse-scale nodes
#             coarse_to_fine_connections: dict mapping coarse node indices to lists of fine node indices
#                                       e.g., {36: [36, 35, 37, 38, 34, 13, 69, 68], ...}
#             fine_graph_edges: [2, E] - edge indices for the fine-scale graph
            
#         Returns:
#             fine_features: [N_fine, feature_dim] - distributed features for fine nodes
#         """
#         N_coarse, feature_dim = coarse_features.shape
#         device = coarse_features.device
        
#         import time
#         t0 = time.time()

#         # Determine N_fine from the connections
#         all_fine_indices = set()
#         for fine_indices in coarse_to_fine_connections.values():
#             all_fine_indices.update(fine_indices)
#         N_fine = max(all_fine_indices) + 1 if all_fine_indices else 0
        
#         print(time.time()-t0)

#         # Initialize fine features
#         fine_features = torch.zeros(N_fine, feature_dim, device=device, dtype=coarse_features.dtype)
        
#         # Create mapping from coarse index in connections to feature index
#         coarse_indices = sorted(coarse_to_fine_connections.keys())
#         coarse_idx_to_feature_idx = {coarse_idx: i for i, coarse_idx in enumerate(coarse_indices)}
        
#         print(time.time()-t0)
#         # First pass: distribute coarse features to their connected fine nodes
#         fine_node_coarse_parents = {}  # Track which coarse nodes influence each fine node
        
#         for coarse_idx, connected_fine_indices in coarse_to_fine_connections.items():
#             coarse_feature_idx = coarse_idx_to_feature_idx[coarse_idx]
#             coarse_feature = coarse_features[coarse_feature_idx]
            
#             for fine_idx in connected_fine_indices:
#                 if fine_idx >= N_fine:
#                     continue
                    
#                 if fine_idx not in fine_node_coarse_parents:
#                     fine_node_coarse_parents[fine_idx] = []
#                 fine_node_coarse_parents[fine_idx].append((coarse_idx, coarse_feature))
        
#         print(time.time()-t0)

#         # Distribute features with weighted averaging if a fine node has multiple coarse parents
#         for fine_idx, coarse_parents in fine_node_coarse_parents.items():
#             if len(coarse_parents) == 1:
#                 # Single parent - direct copy
#                 _, coarse_feature = coarse_parents[0]
#                 fine_features[fine_idx] = coarse_feature
#             else:
#                 # Multiple parents - weighted average (equal weights for now)
#                 feature_sum = torch.zeros_like(coarse_parents[0][1])
#                 for _, coarse_feature in coarse_parents:
#                     feature_sum += coarse_feature
#                 fine_features[fine_idx] = feature_sum / len(coarse_parents)
        
#         print(time.time()-t0)
#         # Second pass: refine using attention over the fine graph structure
#         Q = self.query_proj(fine_features)  # [N_fine, feature_dim]
#         K = self.key_proj(fine_features)    # [N_fine, feature_dim]
#         V = self.value_proj(fine_features)  # [N_fine, feature_dim]
        
#         # Reshape for multi-head attention
#         Q = Q.view(N_fine, self.heads, self.head_dim)
#         K = K.view(N_fine, self.heads, self.head_dim)
#         V = V.view(N_fine, self.heads, self.head_dim)
        
#         # Create adjacency info for efficient attention
#         row, col = fine_graph_edges
#         refined_features = torch.zeros_like(fine_features)
#         print(time.time()-t0)
#         # Only refine nodes that have non-zero features
#         nodes_to_refine = list(fine_node_coarse_parents.keys())
        
#         for node_idx in nodes_to_refine:
#             # Find neighbors of this node
#             neighbors = col[row == node_idx]
#             if len(neighbors) == 0:
#                 refined_features[node_idx] = fine_features[node_idx]
#                 continue
                
#             # Include self in the attention and filter to only nodes with features
#             all_candidates = torch.cat([torch.tensor([node_idx], device=neighbors.device), neighbors])
#             # Keep only nodes that have been assigned features
#             valid_candidates = []
#             for candidate in all_candidates:
#                 if candidate.item() in fine_node_coarse_parents or candidate.item() == node_idx:
#                     valid_candidates.append(candidate)
            
#             if len(valid_candidates) == 0:
#                 refined_features[node_idx] = fine_features[node_idx]
#                 continue
                
#             neighbors = torch.stack(valid_candidates)
            
#             # Limit neighbors for computational efficiency
#             if len(neighbors) > 32:
#                 # Keep self + random subset of neighbors
#                 self_mask = neighbors == node_idx
#                 self_indices = neighbors[self_mask]
#                 other_indices = neighbors[~self_mask]
#                 if len(other_indices) > 31:
#                     perm = torch.randperm(len(other_indices))[:31]
#                     other_indices = other_indices[perm]
#                 neighbors = torch.cat([self_indices, other_indices])
            
#             # Get query for this node and keys/values for neighbors
#             q_node = Q[node_idx:node_idx+1]  # [1, heads, head_dim]
#             k_neighbors = K[neighbors]        # [n_neighbors, heads, head_dim]
#             v_neighbors = V[neighbors]        # [n_neighbors, heads, head_dim]
            
#             # Compute attention scores
#             scores = torch.einsum('qhd,nhd->qhn', q_node, k_neighbors) * self.scale
#             attention_weights = F.softmax(scores, dim=-1)  # [1, heads, n_neighbors]
#             attention_weights = self.dropout_layer(attention_weights)
            
#             # Apply attention to values
#             attended = torch.einsum('qhn,nhd->qhd', attention_weights, v_neighbors)
#             refined_features[node_idx] = attended.view(feature_dim)
#         print(time.time()-t0)
#         # Apply interpolation projection
#         refined_features = self.interpolation_proj(refined_features)
        
#         # Residual connection with original unpooled features
#         final_features = fine_features + refined_features
        
#         return final_features


class SimpleLearnablePooling(nn.Module):
    """
    Simple learnable pooling using sparse matrices.
    """
    def __init__(self, connections_dict: Dict[int, List[int]]):
        super().__init__()
        
        # Convert connections to sparse matrix indices
        coarse_indices = sorted(connections_dict.keys())
        n_coarse = len(coarse_indices)
        
        # Find max fine index to determine matrix size
        all_fine_indices = []
        for fine_list in connections_dict.values():
            all_fine_indices.extend(fine_list)
        n_fine = max(all_fine_indices) + 1 if all_fine_indices else 0
        
        # Create sparse pooling matrix indices
        row_indices = []
        col_indices = []
        
        for output_idx, coarse_idx in enumerate(coarse_indices):
            fine_indices = connections_dict[coarse_idx]
            for fine_idx in fine_indices:
                if fine_idx < n_fine:
                    row_indices.append(output_idx)
                    col_indices.append(fine_idx)
        
        # Store as buffers
        indices = torch.stack([torch.tensor(row_indices), torch.tensor(col_indices)])
        self.register_buffer('indices', indices)
        self.register_buffer('size', torch.tensor([n_coarse, n_fine]))
        
        # Learnable weights for the connections
        self.weights = nn.Parameter(torch.ones(len(row_indices)))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [n_fine, feature_dim]
        returns: [n_coarse, feature_dim]
        """
        # Create sparse matrix
        sparse_matrix = torch.sparse_coo_tensor(
            self.indices, self.weights, self.size.tolist(), device=x.device
        ).coalesce()
        # Apply pooling: sparse_matrix @ x
        with torch.cuda.amp.autocast(enabled=False): 
            x = x.float()
            x = torch.sparse.mm(sparse_matrix, x)
        return x


class SimpleLearnableUnpooling(nn.Module):
    """
    Simple learnable unpooling using sparse matrices.
    """
    def __init__(self, connections_dict: Dict[int, List[int]]):
        super().__init__()
        
        # Convert connections to sparse matrix indices (transpose of pooling)
        coarse_indices = sorted(connections_dict.keys())
        n_coarse = len(coarse_indices)
        
        # Find max fine index
        all_fine_indices = []
        for fine_list in connections_dict.values():
            all_fine_indices.extend(fine_list)
        n_fine = max(all_fine_indices) + 1 if all_fine_indices else 0
        
        # Create sparse unpooling matrix indices (fine x coarse)
        row_indices = []
        col_indices = []
        
        for output_idx, coarse_idx in enumerate(coarse_indices):
            fine_indices = connections_dict[coarse_idx]
            for fine_idx in fine_indices:
                if fine_idx < n_fine:
                    row_indices.append(fine_idx)      # fine nodes in rows
                    col_indices.append(output_idx)    # coarse nodes in cols
        
        # Store as buffers
        indices = torch.stack([torch.tensor(row_indices), torch.tensor(col_indices)])
        self.register_buffer('indices', indices)
        self.register_buffer('size', torch.tensor([n_fine, n_coarse]))
        
        # Learnable weights
        self.weights = nn.Parameter(torch.ones(len(row_indices)))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [n_coarse, feature_dim]
        returns: [n_fine, feature_dim]
        """
        # Create sparse matrix
        sparse_matrix = torch.sparse_coo_tensor(
            self.indices, self.weights, self.size.tolist(), device=x.device
        ).coalesce()
        # Apply unpooling: sparse_matrix @ x
        with torch.cuda.amp.autocast(enabled=False): 
            x = x.float()
            x = torch.sparse.mm(sparse_matrix, x)
        return x



class ResidualBlock(nn.Module):
    """Residual block with dimension matching for progressive architectures."""
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        # Projection layer for dimension mismatch
        if in_dim != out_dim:
            self.projection = nn.Linear(in_dim, out_dim)
        else:
            self.projection = nn.Identity()
            
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x_in: torch.Tensor, x_out: torch.Tensor) -> torch.Tensor:
        """Apply residual connection with optional projection."""
        residual = self.projection(x_in)
        residual = self.dropout(residual)
        return x_out + residual


class ProgressiveEncoder(nn.Module):
    """
    Progressive Encoder with attention-based pooling using explicit hierarchical connections.
    """
    def __init__(self, in_dim: int, hidden_dims: List[int], latent_dim: int,
                 graphs: List[torch.Tensor], level_connections: List[Dict[int, List[int]]], 
                 pooling_schedule: List[int], heads: int = 4, gat_dropout: float = 0.2, 
                 feature_dropout: float = 0.3, latent_dropout: float = 0.1, 
                 latent_noise_std: float = 0.05, use_residuals: bool = True, use_residualsIO: bool = False, log: bool = False):
        """
        Args:
            in_dim: Input feature dimension
            hidden_dims: List of hidden dimensions in decreasing order for progressive compression
            latent_dim: Final latent dimension
            graphs: List of graph edge indices for each hierarchical level
            level_connections: List of dicts where level_connections[i] maps coarse node indices 
                             to lists of fine node indices they are connected to
                             e.g., level_connections[0][36] = [36, 35, 37, 38, 34, 13, 69, 68]
            pooling_schedule: List of layer indices where pooling should occur
            heads: Number of attention heads for GAT layers
            gat_dropout: Dropout rate for GAT layers
            feature_dropout: Dropout rate for feature layers
            latent_dropout: Dropout rate for latent layer
            latent_noise_std: Standard deviation for latent noise during training
            use_residuals: Enable residual connections where appropriate
            use_residualsIO: Enable residuals for input/output layers only
            log: Enable debug logging
        """
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.graphs = graphs
        self.level_connections = level_connections
        self.pooling_schedule = sorted(pooling_schedule)
        self.heads = heads
        self.gat_dropout = gat_dropout
        self.feature_dropout = feature_dropout
        self.latent_dropout = latent_dropout
        self.latent_noise_std = latent_noise_std
        self.log = log
        self.use_residuals = use_residuals
        self.use_residualsIO = use_residualsIO


        if self.log:
            print("=" * 60)
            print("PROGRESSIVE ENCODER INITIALIZATION (Attention-based with level_connections)")
            print("=" * 60)
            print(f"Input dim: {in_dim}")
            print(f"Hidden dims: {hidden_dims}")
            print(f"Latent dim: {latent_dim}")
            print(f"Pooling schedule: {pooling_schedule}")
            print(f"Level connections: {len(level_connections) if level_connections else 0} levels")

        # Build the progressive architecture
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.residuals = nn.ModuleList()
        self.attention_poolers = nn.ModuleDict()
        
        # Create dimensions list: [in_dim] + hidden_dims + [latent_dim]
        all_dims = [in_dim] + hidden_dims + [latent_dim]

        # Create attention pooling modules for scheduled layers
        self.poolers = nn.ModuleDict()
        for pool_step, layer_idx in enumerate(pooling_schedule):
            if layer_idx >= len(all_dims) - 1:
                raise ValueError(f"Pooling scheduled for layer {layer_idx} but only {len(all_dims)-1} layers exist")
            out_features = all_dims[layer_idx + 1]
            connections = level_connections[pool_step]
            self.poolers[str(layer_idx)] = SimpleLearnablePooling(connections)

            # self.attention_poolers[str(layer_idx)] = AttentionPooling(
            #     out_features, heads=heads, dropout=gat_dropout
            # )
            if self.log:
                print(f"Created attention pooler for layer {layer_idx} with {out_features} features")
        
        for i in range(len(all_dims) - 1):
            in_features = all_dims[i]
            out_features = all_dims[i + 1]
            
            if self.log: 
                print(f"Building layer {i}: {in_features} -> {out_features}")
            
            if i == 0:
                # First layer: direct projection
                self.layers.append(nn.Linear(in_features, out_features))
                if use_residualsIO:
                    self.residuals.append(ResidualBlock(in_features, out_features, feature_dropout * 0.5))
                    if self.log: print(f"  Added residual block for first layer")
                else:
                    self.residuals.append(None)
                    
            elif i == len(all_dims) - 2:
                # Last layer: project to latent with intermediate expansion
                intermediate_dim = max(latent_dim * 2, out_features)
                self.layers.append(nn.Sequential(
                    nn.Linear(in_features, intermediate_dim),
                    nn.GELU(approximate='tanh'),
                    nn.Dropout(feature_dropout),
                    nn.Linear(intermediate_dim, out_features)
                ))
                self.residuals.append(None)
                if self.log: print(f"  Created latent projection with intermediate dim {intermediate_dim}")
                
            else:
                # Middle layers: GAT
                gat_out_per_head = out_features // heads
                self.layers.append(GATv2Conv(in_features, gat_out_per_head, 
                                            heads=heads, concat=True, dropout=gat_dropout,
                                            add_self_loops=True, bias=True, residual=use_residuals))
                self.residuals.append(None)
                if self.log: print(f"  Created GAT layer with {heads} heads, {gat_out_per_head} out per head")
            
            # Add normalization for all but the last layer
            if i < len(all_dims) - 2:
                # num_groups = min(8, out_features // 4)
                # if num_groups > 0 and out_features % num_groups == 0:
                #     self.norms.append(DiffGroupNorm(out_features, num_groups))
                #     if self.log: print(f"  Added DiffGroupNorm with {num_groups} groups")
                # else:
                self.norms.append(LayerNorm(out_features))
                #     if self.log: print(f"  Added LayerNorm (fallback)")

        if self.log:
            print(f"Total encoder layers: {len(self.layers)}")
            print("=" * 60)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.log: 
            print("\n" + "=" * 40)
            print("ENCODER FORWARD PASS (Attention-based with level_connections)")
            print("=" * 40)
            print(f"Encoder input: {x.shape}")
        
        current_graph_level = 0
        pooling_step = 0

        for i, layer in enumerate(self.layers):
            if self.log:
                print(f"\n--- Layer {i} ---")
                print(f"Input shape: {x.shape}")
                print(f"Current graph level: {current_graph_level}")
                
            x_input = x
            
            if i == 0:
                # First layer: linear projection
                x = layer(x)
                x = self.norms[i](x)
                x = F.gelu(x, approximate='tanh')
                x = F.dropout(x, p=self.feature_dropout, training=self.training)
                
                # Apply residual connection if enabled
                if self.residuals[i] is not None:
                    x = self.residuals[i](x_input, x)
                    if self.log: print(f"Applied residual connection")
                    
            elif i == len(self.layers) - 1:
                # Last layer: to latent space
                x = layer(x)
                if self.training:
                    x = F.dropout(x, p=self.latent_dropout)
                    if self.latent_noise_std > 0:
                        noise = torch.randn_like(x) * self.latent_noise_std
                        x = x + noise
                        if self.log: print(f"Added latent noise with std {self.latent_noise_std}")

            else:
                # Middle GAT layers
                edge_idx = self.graphs[current_graph_level]
                if self.log:
                    print(f"Using graph level {current_graph_level} with {edge_idx.shape[1]} edges")
                
                def create_layer_forward(gat_layer, norm_layer, edge_idx, feature_dropout):
                    def layer_forward(x_in):
                        x_out = gat_layer(x_in, edge_idx)
                        x_out = norm_layer(x_out)
                        x_out = F.gelu(x_out, approximate='tanh')
                        x_out = F.dropout(x_out, p=feature_dropout, training=self.training)
                        return x_out
                    return layer_forward
                
                layer_fn = create_layer_forward(layer, self.norms[i], edge_idx, self.feature_dropout)
                x = checkpoint(layer_fn, x, use_reentrant=False)

            # Check if attention pooling should occur after this layer
            if i in self.pooling_schedule and pooling_step < len(self.level_connections):
                if self.log:
                    print(f"Applying attention pooling step {pooling_step} after layer {i}")
                    print(f"Before pooling: {x.shape}")
                
                if self.log:
                    print(f"Using level_connections[{pooling_step}] with {len(self.level_connections[pooling_step])} coarse nodes")

                # Apply attention pooling
                # pooler = self.attention_poolers[str(i)]
                pooler = self.poolers[str(i)]
                x = pooler(x)

                if self.log: 
                    print(f"Applied attention pooling, new shape: {x.shape}")
                
                pooling_step += 1
                current_graph_level += 1
                
                if self.log:
                    print(f"Updated to graph level {current_graph_level}, pooling step {pooling_step}")
            
            if self.log: 
                print(f"After layer {i}: {x.shape}")
                if x.numel() > 0:
                    print(f"Output stats - min: {x.min():.4f}, max: {x.max():.4f}, mean: {x.mean():.4f}")
        
        if self.log:
            print(f"\nEncoder final output: {x.shape}")
            print("=" * 40)
        
        return x



class ProgressiveDecoder(nn.Module):
    """
    Progressive Decoder with attention-based unpooling using explicit hierarchical connections.
    """
    def __init__(self, latent_dim: int, hidden_dims: List[int], out_dim: int,
                 graphs: List[torch.Tensor], level_connections: List[Dict[int, List[int]]], 
                 unpooling_schedule: List[int], heads: int = 4, gat_dropout: float = 0.2, 
                 feature_dropout: float = 0.3, 
                 use_residuals: bool = True, use_residualsIO: bool = False, log: bool = False):
        """
        Args:
            latent_dim: Input latent dimension
            hidden_dims: List of hidden dimensions in increasing order for progressive expansion
            out_dim: Final output dimension
            graphs: List of graph edge indices for each hierarchical level (from coarse to fine)
            level_connections: List of dicts where level_connections[i] maps coarse node indices 
                             to lists of fine node indices they are connected to
                             e.g., level_connections[0][36] = [36, 35, 37, 38, 34, 13, 69, 68]
            unpooling_schedule: List of layer indices where unpooling should occur
            heads: Number of attention heads for GAT layers
            gat_dropout: Dropout rate for GAT layers
            feature_dropout: Dropout rate for feature layers
            use_residuals: Enable residual connections where appropriate
            use_residualsIO: Enable residuals for input/output layers only
            log: Enable debug logging
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.out_dim = out_dim
        self.graphs = graphs
        self.level_connections = level_connections
        self.unpooling_schedule = sorted(unpooling_schedule)
        self.heads = heads
        self.gat_dropout = gat_dropout
        self.feature_dropout = feature_dropout
        self.log = log
        self.use_residuals = use_residuals
        self.use_residualsIO = use_residualsIO

        if self.log:
            print("=" * 60)
            print("PROGRESSIVE DECODER INITIALIZATION (Attention-based with level_connections)")
            print("=" * 60)
            print(f"Latent dim: {latent_dim}")
            print(f"Hidden dims: {hidden_dims}")
            print(f"Output dim: {out_dim}")
            print(f"Unpooling schedule: {unpooling_schedule}")
            print(f"Level connections: {len(level_connections) if level_connections else 0} levels")

        # Build the progressive architecture
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.residuals = nn.ModuleList()
        self.attention_unpoolers = nn.ModuleDict()
        
        # Create dimensions list: [latent_dim] + hidden_dims + [out_dim]
        all_dims = [latent_dim] + hidden_dims + [out_dim]

        # Create attention unpooling modules for scheduled layers
        for unpool_step, layer_idx in enumerate(unpooling_schedule):
            if layer_idx >= len(all_dims) - 1:
                raise ValueError(f"Unpooling scheduled for layer {layer_idx} but only {len(all_dims)-1} layers exist")
            in_features = all_dims[layer_idx]

            connections = level_connections[unpool_step]
            self.attention_unpoolers[str(layer_idx)] = SimpleLearnableUnpooling(connections)
            # self.attention_unpoolers[str(layer_idx)] = AttentionUnpooling(
            #     in_features, heads=heads, dropout=gat_dropout
            # )
            if self.log:
                print(f"Created attention unpooler for layer {layer_idx} with {in_features} features")

        for i in range(len(all_dims) - 1):
            in_features = all_dims[i]
            out_features = all_dims[i + 1]
            
            if self.log: 
                print(f"Building layer {i}: {in_features} -> {out_features}")
            
            if i == 0:
                # First layer: from latent space with enhanced expansion
                intermediate_dim = max(latent_dim * 2, out_features)
                self.layers.append(nn.Sequential(
                    nn.Linear(in_features, intermediate_dim),
                    nn.GELU(approximate='tanh'),
                    nn.Dropout(feature_dropout),
                    nn.Linear(intermediate_dim, out_features)
                ))
                if self.log: print(f"  Created latent expansion with intermediate dim {intermediate_dim}")
                
                # self.layers.append(nn.Linear(in_features, out_features))
                self.residuals.append(None)
                
            elif i == len(all_dims) - 2:
                # Last layer: to output with detail preservation
                self.layers.append(nn.Sequential(
                    nn.Linear(in_features, in_features),
                    nn.GELU(approximate='tanh'),
                    nn.Linear(in_features, out_features)
                ))
                if self.log: print(f"  Created two-stage output layer")
                # self.layers.append(nn.Linear(in_features, out_features))
                
                # Residual connection for final reconstruction layer
                if use_residualsIO:
                    self.residuals.append(ResidualBlock(in_features, out_features, feature_dropout * 0.5))
                    if self.log: print(f"  Added residual block for final layer")
                else:
                    self.residuals.append(None)
                    
            else:
                # Middle layers: GAT with enhanced reconstruction
                gat_out_per_head = out_features // heads
                self.layers.append(GATv2Conv(in_features, gat_out_per_head,
                                            heads=heads, concat=True, dropout=gat_dropout,
                                            add_self_loops=True, bias=True, residual=use_residuals))
                self.residuals.append(None)
  
            # Add normalization for all but the last layer
            if i < len(all_dims) - 2:
                # num_groups = min(8, out_features // 4)
                # if num_groups > 0 and out_features % num_groups == 0:
                #     self.norms.append(DiffGroupNorm(out_features, num_groups))
                #     if self.log: print(f"  Added DiffGroupNorm with {num_groups} groups")
                # else:
                self.norms.append(LayerNorm(out_features))
                #     if self.log: print(f"  Added LayerNorm (fallback)")

        if self.log:
            print(f"Total decoder layers: {len(self.layers)}")
            print("=" * 60)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        if self.log: 
            print("\n" + "=" * 40)
            print("DECODER FORWARD PASS (Attention-based with level_connections)")
            print("=" * 40)
            print(f"Decoder input: {latent.shape}")
        
        # Start at coarsest graph level and work towards finest
        current_graph_level = len(self.graphs) - 1
        unpooling_step = len(self.unpooling_schedule) - 1

        if self.log:
            print(f"Starting at graph level {current_graph_level}")
            print(f"Starting unpooling step {unpooling_step}")

        x = latent
        for i, layer in enumerate(self.layers):
            if self.log:
                print(f"\n--- Layer {i} ---")
                print(f"Input shape: {x.shape}")
                print(f"Current graph level: {current_graph_level}")
                
            x_input = x
            
            # Check if attention unpooling should occur before this layer
            if i in self.unpooling_schedule and unpooling_step >= 0:
                if self.log:
                    print(f"Applying attention unpooling step {unpooling_step} before layer {i}")
                    print(f"Before unpooling: {x.shape}")

                if self.log:
                    print(f"Using level_connections[{unpooling_step}] with {len(self.level_connections[unpooling_step])} coarse nodes")
                    print(f"Using fine graph level {current_graph_level - 1 if current_graph_level > 0 else 0}")

                # Apply attention unpooling
                # unpooler = self.attention_unpoolers[str(i)]
                # x = unpooler(x, connections, fine_graph)
                unpooler = self.attention_unpoolers[str(i)]
                x = unpooler(x)

                if self.log: 
                    print(f"Applied attention unpooling, new shape: {x.shape}")
                
                unpooling_step -= 1
                current_graph_level -= 1
                
                if self.log:
                    print(f"Updated to graph level {current_graph_level}, unpooling step {unpooling_step}")

            if i == 0:
                # First layer: from latent
                x = layer(x)
                x = self.norms[i](x)
                x = F.gelu(x, approximate='tanh')
                x = F.dropout(x, p=self.feature_dropout, training=self.training)
                
            elif i == len(self.layers) - 1:
                # Last layer: to output (no activation for final reconstruction)
                x = layer(x)
                if self.residuals[i] is not None:
                    x = self.residuals[i](x_input, x)
                    if self.log: print(f"Applied final residual connection")
                    
            else:
                # Middle GAT layers
                edge_idx = self.graphs[current_graph_level]
                if self.log:
                    print(f"Using graph level {current_graph_level} with {edge_idx.shape[1]} edges")

                def create_layer_forward(gat_layer, norm_layer, edge_idx, feature_dropout):
                    def layer_forward(x_in):
                        x_out = gat_layer(x_in, edge_idx)
                        x_out = norm_layer(x_out)
                        x_out = F.gelu(x_out, approximate='tanh')
                        x_out = F.dropout(x_out, p=feature_dropout, training=self.training)
                        return x_out
                    return layer_forward
                
                layer_fn = create_layer_forward(layer, self.norms[i], 
                                               edge_idx, self.feature_dropout)
                x = checkpoint(layer_fn, x, use_reentrant=False)
            
            if self.log: 
                print(f"After layer {i}: {x.shape}")
                if x.numel() > 0:
                    print(f"Output stats - min: {x.min():.4f}, max: {x.max():.4f}, mean: {x.mean():.4f}")
        
        if self.log:
            print(f"\nDecoder final output: {x.shape}")
            print("=" * 40)
        
        return x



import torch
import torch.nn as nn
from typing import List, Dict

class ProgressiveGraphAutoencoder(nn.Module):
    """
    Graph Autoencoder with attention-based progressive compression/expansion using explicit hierarchical connections.
    
    This autoencoder uses your level_connections data structure to perform proper hierarchical pooling
    and unpooling, where each coarse node attends to its explicitly connected fine-scale nodes.
    """
    def __init__(self, in_dim: int, latent_dim: int = 2, out_dim: int = 3,
                 encoder_hidden_dims: List[int] = None, 
                 decoder_hidden_dims: List[int] = None,
                 graphs: List[torch.Tensor] = None,
                 level_connections: List[Dict[int, List[int]]] = None,
                 pooling_schedule: List[int] = None,
                 unpooling_schedule: List[int] = None,
                 heads: int = 4, gat_dropout: float = 0.2, feature_dropout: float = 0.3, 
                 latent_dropout: float = 0.1, latent_noise_std: float = 0.05,
                 use_residuals: bool = True, 
                 use_residualsIO: bool = False, log: bool = False):
        """
        Initialize the Progressive Graph Autoencoder with attention-based pooling/unpooling.
        
        Args:
            in_dim: Input feature dimension
            latent_dim: Latent space dimension
            out_dim: Output feature dimension
            encoder_hidden_dims: Hidden dimensions for encoder layers (decreasing order)
            decoder_hidden_dims: Hidden dimensions for decoder layers (increasing order)
            graphs: List of graph edge indices for each hierarchical level
                   graphs[0] = finest level, graphs[-1] = coarsest level
            level_connections: List of dicts where level_connections[i] maps coarse node indices 
                             to lists of fine node indices they are connected to
                             e.g., level_connections[0][36] = [36, 35, 37, 38, 34, 13, 69, 68]
                             level_connections[0] maps from level 1 to level 0 (fine to coarse)
            pooling_schedule: List of layer indices where pooling should occur in encoder
                            e.g., [2, 4] means pool after layers 2 and 4
            unpooling_schedule: List of layer indices where unpooling should occur in decoder
                              e.g., [1, 3] means unpool before layers 1 and 3
            heads: Number of attention heads for GAT layers
            gat_dropout: Dropout rate for GAT layers
            feature_dropout: Dropout rate for feature layers
            latent_dropout: Dropout rate for latent layer
            latent_noise_std: Standard deviation for latent noise during training
            use_residuals: Enable residual connections in GAT layers
            use_residualsIO: Enable residuals for input/output layers only
            log: Enable debug logging
            
        Example usage:
            ```python
            # Your data from build_hierarchical_graph_v6
            graphs, node_mappings, node_counts, level_edges, level_connections = build_hierarchical_graph_v6(...)
            
            model = ProgressiveGraphAutoencoder(
                in_dim=your_input_features.shape[1],
                latent_dim=32,
                out_dim=3,  # e.g., for 3D coordinates
                graphs=graphs,
                level_connections=level_connections,
                pooling_schedule=[2, 4],      # Pool after layers 2 and 4
                unpooling_schedule=[1, 3],    # Unpool before layers 1 and 3
                heads=8,
                log=True
            )
            
            # Forward pass
            reconstructed = model(input_features)
            reconstructed, latent = model(input_features, return_latent=True)
            ```
        """

        super().__init__()
        
        # Set default progressive dimensions if not provided
        if encoder_hidden_dims is None:
            # Default: progressive compression
            encoder_hidden_dims = [min(512, in_dim * 2), 256, 128, 64]
        if decoder_hidden_dims is None:
            # Default: progressive expansion (should mirror encoder in reverse)
            decoder_hidden_dims = [64, 128, 256, min(512, out_dim * 2)]
        
        # Set default pooling schedules if not provided
        if pooling_schedule is None:
            # Default: pool after every other GAT layer (skip first and last layers)
            pooling_schedule = list(range(2, len(encoder_hidden_dims), 2))
        if unpooling_schedule is None:
            # Default: unpool before every other GAT layer (skip first and last layers)
            unpooling_schedule = list(range(1, len(decoder_hidden_dims), 2))
        
        # Store configuration
        self.in_dim = in_dim
        self.latent_dim = latent_dim
        self.out_dim = out_dim
        self.graphs = graphs
        self.level_connections = level_connections
        self.pooling_schedule = pooling_schedule
        self.unpooling_schedule = unpooling_schedule
        self.latent_dropout = latent_dropout
        self.latent_noise_std = latent_noise_std
        self.log = log
        self.use_residuals = use_residuals

        # Validation checks
        if graphs is None:
            raise ValueError("graphs must be provided")
        if level_connections is None:
            raise ValueError("level_connections must be provided")
        
        # Ensure dimensions are compatible with heads
        encoder_hidden_dims = [max(d, heads) for d in encoder_hidden_dims]
        decoder_hidden_dims = [max(d, heads) for d in decoder_hidden_dims]
        
        # Trim level_connections to match schedule length
        if len(level_connections) > len(pooling_schedule):
            level_connections = level_connections[:len(pooling_schedule)]
            if log:
                print(f"Trimmed level_connections to {len(level_connections)} to match pooling schedule")

        # Trim graphs to match the number of levels needed
        if len(graphs) > len(pooling_schedule) + 1:
            graphs = graphs[:len(pooling_schedule) + 1]
            if log:
                print(f"Trimmed graphs to {len(graphs)} to match pooling levels")

        # Validate schedules
        if len(pooling_schedule) != len(level_connections):
            raise ValueError(f"pooling_schedule length ({len(pooling_schedule)}) must match "
                           f"level_connections length ({len(level_connections)})")
        
        if len(unpooling_schedule) != len(level_connections):
            print(f"Warning: unpooling_schedule length ({len(unpooling_schedule)}) doesn't match "
                  f"level_connections length ({len(level_connections)}). "
                  f"Trimming unpooling_schedule to match.")
            unpooling_schedule = unpooling_schedule[:len(level_connections)]

        

        # Initialize encoder and decoder
        self.encoder = ProgressiveEncoder(
            in_dim=in_dim, 
            hidden_dims=encoder_hidden_dims, 
            latent_dim=latent_dim, 
            graphs=graphs, 
            level_connections=level_connections, 
            pooling_schedule=pooling_schedule,
            heads=heads, 
            gat_dropout=gat_dropout, 
            feature_dropout=feature_dropout, 
            latent_dropout=latent_dropout, 
            latent_noise_std=latent_noise_std, 
            use_residuals=use_residuals, 
            use_residualsIO=use_residualsIO, 
            log=log
        )
        
        self.decoder = ProgressiveDecoder(
            latent_dim=latent_dim, 
            hidden_dims=decoder_hidden_dims, 
            out_dim=out_dim, 
            graphs=graphs, 
            level_connections=level_connections[::-1], 
            unpooling_schedule=unpooling_schedule,
            heads=heads, 
            gat_dropout=gat_dropout, 
            feature_dropout=feature_dropout, 
            use_residuals=use_residuals, 
            use_residualsIO=use_residualsIO, 
            log=log
        )
        
        if log:
            print("\n" + "=" * 80)
            print("PROGRESSIVE GRAPH AUTOENCODER INITIALIZED")
            print("=" * 80)
            print(f"Architecture:")
            print(f"  Input: {in_dim} -> Encoder: {encoder_hidden_dims} -> Latent: {latent_dim}")
            print(f"  Latent: {latent_dim} -> Decoder: {decoder_hidden_dims} -> Output: {out_dim}")
            print(f"Hierarchical Structure:")
            print(f"  Number of graph levels: {len(graphs)}")
            print(f"  Number of connection levels: {len(level_connections)}")
            print(f"  Pooling schedule: {pooling_schedule}")
            print(f"  Unpooling schedule: {unpooling_schedule}")
            print(f"Training Configuration:")
            print(f"  GAT heads: {heads}")
            print(f"  GAT dropout rate: {gat_dropout}")
            print(f"  Feature dropout rate: {feature_dropout}")
            print(f"  Latent dropout rate: {latent_dropout}")
            print(f"  Latent noise std: {latent_noise_std}")
            print(f"Model Features:")
            print(f"  Residual connections: {use_residuals}")
            print(f"  Residual connections (IO): {use_residualsIO}")
            
            # Print some statistics about the hierarchical structure
            if level_connections:
                for i, connections in enumerate(level_connections):
                    total_fine_nodes = sum(len(fine_list) for fine_list in connections.values())
                    avg_connections = total_fine_nodes / len(connections) if connections else 0
                    print(f"  Level {i}: {len(connections)} coarse nodes, "
                          f"avg {avg_connections:.1f} fine connections per coarse node")
            print("=" * 80)

    def forward(self, x: torch.Tensor, return_latent: bool = False):
        """
        Forward pass through the progressive autoencoder.
        
        Args:
            x: Input node features [N_nodes, in_dim]
            return_latent: If True, returns (reconstructed, latent), else just reconstructed
            
        Returns:
            reconstructed: Reconstructed node features [N_nodes, out_dim] 
            latent: Latent representation [N_latent_nodes, latent_dim] (if return_latent=True)
        """
        if self.log:
            print(f"\n{'='*50}")
            print(f"AUTOENCODER FORWARD PASS")
            print(f"{'='*50}")
            print(f"Input shape: {x.shape}")
        
        # Encode to latent space
        latent = self.encoder(x)
        
        if self.log:
            print(f"Latent shape: {latent.shape}")
        
        # Decode from latent space
        reconstructed = self.decoder(latent)
        
        if self.log:
            print(f"Reconstructed shape: {reconstructed.shape}")
            print(f"{'='*50}")
        
        if return_latent:
            return reconstructed, latent
        return reconstructed

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to latent space.
        
        Args:
            x: Input node features [N_nodes, in_dim]
            
        Returns:
            latent: Latent representation [N_latent_nodes, latent_dim]
        """
        return self.encoder(x)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode from latent space to output.
        
        Args:
            latent: Latent representation [N_latent_nodes, latent_dim]
            
        Returns:
            reconstructed: Reconstructed node features [N_nodes, out_dim]
        """
        return self.decoder(latent)
    
    def get_compression_ratio(self) -> float:
        """
        Calculate the compression ratio achieved by the autoencoder.
        
        Returns:
            ratio: Input size / Latent size
        """
        if not self.graphs or not self.level_connections:
            return 1.0
            
        input_nodes = self.graphs[0].max().item() + 1  # Assumes 0-indexed nodes
        
        # Find latent nodes from the final level connections
        final_connections = self.level_connections[-1]
        latent_nodes = len(final_connections)
        
        return input_nodes / latent_nodes if latent_nodes > 0 else 1.0
    
    def print_architecture_summary(self):
        """Print a detailed summary of the model architecture."""
        print("\n" + "=" * 80)
        print("PROGRESSIVE GRAPH AUTOENCODER ARCHITECTURE SUMMARY")
        print("=" * 80)
        
        # Calculate total parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"Total Parameters: {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")
        print(f"Compression Ratio: {self.get_compression_ratio():.2f}x")
        
        print(f"\nEncoder Architecture:")
        for i, layer in enumerate(self.encoder.layers):
            layer_type = type(layer).__name__
            if hasattr(layer, 'in_features') and hasattr(layer, 'out_features'):
                print(f"  Layer {i}: {layer_type} ({layer.in_features} -> {layer.out_features})")
            elif hasattr(layer, 'in_channels') and hasattr(layer, 'out_channels'):
                print(f"  Layer {i}: {layer_type} ({layer.in_channels} -> {layer.out_channels})")
            else:
                print(f"  Layer {i}: {layer_type}")
        
        print(f"\nDecoder Architecture:")
        for i, layer in enumerate(self.decoder.layers):
            layer_type = type(layer).__name__
            if hasattr(layer, 'in_features') and hasattr(layer, 'out_features'):
                print(f"  Layer {i}: {layer_type} ({layer.in_features} -> {layer.out_features})")
            elif hasattr(layer, 'in_channels') and hasattr(layer, 'out_channels'):
                print(f"  Layer {i}: {layer_type} ({layer.in_channels} -> {layer.out_channels})")
            else:
                print(f"  Layer {i}: {layer_type}")
        
        print("=" * 80)