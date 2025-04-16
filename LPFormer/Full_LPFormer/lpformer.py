import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import GCNConv
from torch_geometric.utils import add_self_loops, degree
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import scipy.sparse as sp
import networkx as nx
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

#####################################################################
# DATA LOADING AND PREPROCESSING
#####################################################################

def load_marvel_data():
    """
    Load the Marvel hero network data.
    """
    # Load nodes data
    nodes_df = pd.read_csv('nodes.csv')
    
    # Load edges data
    edges_df = pd.read_csv('edges.csv')
    
    # Load hero network data
    hero_edges_df = pd.read_csv('hero-network.csv')
    
    print(f"Nodes: {len(nodes_df)}")
    print(f"Comic-Hero edges: {len(edges_df)}")
    print(f"Hero-Hero edges: {len(hero_edges_df)}")
    
    return nodes_df, edges_df, hero_edges_df

def preprocess_data(nodes_df, edges_df, hero_edges_df):
    """
    Preprocess the Marvel data to create a graph for link prediction.
    """
    # Create a networkx graph from hero edges
    G = nx.from_pandas_edgelist(hero_edges_df, 'hero1', 'hero2')
    
    # Extract hero nodes
    hero_nodes = nodes_df[nodes_df['type'] == 'hero']['node'].tolist()
    
    # Filter heroes that are in the hero_edges_df
    heroes_in_network = set(hero_edges_df['hero1']).union(set(hero_edges_df['hero2']))
    hero_nodes = [h for h in hero_nodes if h in heroes_in_network]
    
    # Create node mapping
    node_mapping = {node: i for i, node in enumerate(hero_nodes)}
    
    # Create edge index
    edge_index = []
    for _, row in hero_edges_df.iterrows():
        if row['hero1'] in node_mapping and row['hero2'] in node_mapping:
            edge_index.append([node_mapping[row['hero1']], node_mapping[row['hero2']]])
            edge_index.append([node_mapping[row['hero2']], node_mapping[row['hero1']]])  # Add reverse edge
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t()
    
    # Create node features based on comics they appear in
    comics = edges_df['comic'].unique()
    comics_map = {comic: i for i, comic in enumerate(comics)}
    
    # Initialize node features matrix
    num_heroes = len(hero_nodes)
    num_comics = len(comics)
    node_features = torch.zeros(num_heroes, num_comics)
    
    # Fill node features matrix
    for _, row in edges_df.iterrows():
        if row['hero'] in node_mapping and row['comic'] in comics_map:
            hero_idx = node_mapping[row['hero']]
            comic_idx = comics_map[row['comic']]
            node_features[hero_idx, comic_idx] = 1.0
    
    # If a hero has no comic appearances, give them a small constant feature
    zero_rows = (node_features.sum(dim=1) == 0)
    if zero_rows.any():
        node_features[zero_rows, :] = 0.01
    
    # Normalize features
    row_sums = node_features.sum(dim=1, keepdim=True)
    node_features = node_features / row_sums.clamp(min=1)
    
    # Create edge data
    edge_data = {
        'edge_index': edge_index,
        'num_nodes': num_heroes,
        'node_features': node_features,
        'node_mapping': node_mapping,
        'reverse_mapping': {i: node for node, i in node_mapping.items()}
    }
    
    return edge_data, G

def split_edges(edge_data, test_ratio=0.15, val_ratio=0.05):
    """
    Split the edges into train, validation, and test sets.
    """
    edge_index = edge_data['edge_index']
    num_nodes = edge_data['num_nodes']
    
    # Get unique undirected edges (remove duplicates)
    edge_list = edge_index.t().tolist()
    undirected_edges = set()
    for edge in edge_list:
        if tuple(edge) not in undirected_edges and tuple(reversed(edge)) not in undirected_edges:
            undirected_edges.add(tuple(edge))
    
    # Convert back to list
    undirected_edges = list(undirected_edges)
    
    # Split edges
    train_edges, test_edges = train_test_split(undirected_edges, test_size=test_ratio + val_ratio)
    val_edges, test_edges = train_test_split(test_edges, test_size=test_ratio/(test_ratio + val_ratio))
    
    # Create negative samples for validation and testing
    all_edge_set = set([tuple(e) for e in undirected_edges])
    all_edge_set_reversed = set([tuple(reversed(e)) for e in undirected_edges])
    all_edges_combined = all_edge_set.union(all_edge_set_reversed)
    
    val_neg_edges = []
    test_neg_edges = []
    
    # For each positive edge, sample a corresponding negative edge
    for edge_set, neg_edge_set in [(val_edges, val_neg_edges), (test_edges, test_neg_edges)]:
        for _ in edge_set:
            while True:
                i, j = np.random.randint(0, num_nodes, 2)
                if i == j:
                    continue
                if (i, j) not in all_edges_combined and (j, i) not in all_edges_combined:
                    neg_edge_set.append((i, j))
                    break
    
    # Create train edge index
    train_edge_index = []
    for edge in train_edges:
        train_edge_index.append(list(edge))
        train_edge_index.append(list(reversed(edge)))  # Add reverse edge
    
    train_edge_index = torch.tensor(train_edge_index, dtype=torch.long).t()
    
    # Convert edge lists to torch tensors
    val_pos_edge_index = torch.tensor(val_edges, dtype=torch.long).t()
    val_neg_edge_index = torch.tensor(val_neg_edges, dtype=torch.long).t()
    test_pos_edge_index = torch.tensor(test_edges, dtype=torch.long).t()
    test_neg_edge_index = torch.tensor(test_neg_edges, dtype=torch.long).t()
    
    return {
        'train_edge_index': train_edge_index,
        'val_pos_edge_index': val_pos_edge_index,
        'val_neg_edge_index': val_neg_edge_index,
        'test_pos_edge_index': test_pos_edge_index,
        'test_neg_edge_index': test_neg_edge_index
    }

#####################################################################
# PPR COMPUTATION
#####################################################################

def compute_ppr_matrix(edge_index, num_nodes, alpha=0.15, eps=1e-5, max_iter=1000):
    """
    Compute the personalized PageRank matrix using power iteration method.
    
    Args:
        edge_index: Edge index tensor of shape [2, num_edges].
        num_nodes: Number of nodes.
        alpha: Teleportation probability.
        eps: Convergence threshold.
        max_iter: Maximum number of iterations.
        
    Returns:
        PPR matrix as a sparse tensor.
    """
    # Create adjacency matrix
    row, col = edge_index
    
    # Create sparse adjacency matrix
    adj = sp.coo_matrix((np.ones(row.size(0)), (row.numpy(), col.numpy())), 
                        shape=(num_nodes, num_nodes))
    
    # Symmetrically normalize adjacency matrix
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)

    # Compute PPR for each node using power iteration
    ppr_matrix = []
    for i in tqdm(range(num_nodes), desc="Computing PPR matrix"):
        # Initialize preference vector (teleport to node i)
        x = np.zeros(num_nodes)
        x[i] = 1.0
        
        # Power iteration
        for _ in range(max_iter):
            old_x = x.copy()
            # Apply random walk with restarts
            x = (1 - alpha) * normalized_adj.dot(x) + alpha * (i == np.arange(num_nodes))
            
            # Check convergence
            if np.linalg.norm(x - old_x, 1) < eps:
                break
        
        # Store the PPR vector for node i
        ppr_matrix.append(x)
    
    # Stack all PPR vectors
    ppr_matrix = np.stack(ppr_matrix)
    
    # Create sparse tensor
    ppr_sparse = sp.csr_matrix(ppr_matrix)
    
    return ppr_sparse

def threshold_ppr_matrix(ppr_matrix, cn_threshold=1e-2, one_hop_threshold=1e-4, multi_hop_threshold=1e-3):
    """
    Apply thresholds to the PPR matrix for different node types.
    """
    # Apply thresholds
    # This will be used later to filter nodes based on their PPR scores
    return {
        'cn_threshold': cn_threshold,
        'one_hop_threshold': one_hop_threshold,
        'multi_hop_threshold': multi_hop_threshold,
        'ppr_matrix': ppr_matrix
    }

#####################################################################
# LPFORMER ARCHITECTURE
#####################################################################

class GATv2Attention(nn.Module):
    def __init__(self, in_features, hidden_dim):
        """
        GATv2 attention mechanism as used in LPFormer.
        
        Args:
            in_features: Dimension of input features.
            hidden_dim: Dimension of hidden features.
        """
        super(GATv2Attention, self).__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        
        # Project node features
        self.W = nn.Linear(in_features, hidden_dim, bias=False)
        
        # Attention mechanism
        self.a = nn.Linear(hidden_dim * 4, 1, bias=False)
        
        # LeakyReLU
        self.leakyrelu = nn.LeakyReLU(0.2)
        
    def forward(self, h_a, h_b, h_u, rpe):
        """
        Compute attention weights.
        
        Args:
            h_a: Features of node a in the target link.
            h_b: Features of node b in the target link.
            h_u: Features of node u being attended to.
            rpe: Relative positional encoding of node u to (a, b).
            
        Returns:
            Attention weights.
        """
        # Project features
        Wh_a = self.W(h_a)
        Wh_b = self.W(h_b)
        Wh_u = self.W(h_u)
        
        # Concatenate features for attention computation
        # Shape: [batch_size, 3*hidden_dim + rpe_dim]
        cat_features = torch.cat([Wh_a, Wh_b, Wh_u, rpe], dim=-1)
        
        # Compute attention scores
        # Shape: [batch_size, 1]
        e = self.a(cat_features)
        
        # Apply LeakyReLU
        return self.leakyrelu(e)

class RelativePositionalEncoding(nn.Module):
    def __init__(self, input_dim, output_dim):
        """
        Relative positional encoding model based on PPR.
        
        Args:
            input_dim: Dimension of input PPR scores.
            output_dim: Dimension of output RPE.
        """
        super(RelativePositionalEncoding, self).__init__()
        
        # MLP for CN nodes
        self.mlp_cn = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )
        
        # MLP for 1-hop nodes
        self.mlp_one_hop = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )
        
        # MLP for >1-hop nodes
        self.mlp_multi_hop = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )
    
    def forward(self, ppr_scores, node_type):
        """
        Compute RPE based on PPR scores and node type.
        
        Args:
            ppr_scores: PPR scores for nodes (a,u) and (b,u).
            node_type: Type of node (CN, 1-hop, or >1-hop).
            
        Returns:
            Relative positional encoding.
        """
        # Apply MLP based on node type
        if node_type == 'cn':
            rpe_ab = self.mlp_cn(ppr_scores)
        elif node_type == 'one_hop':
            rpe_ab = self.mlp_one_hop(ppr_scores)
        else:  # multi_hop
            rpe_ab = self.mlp_multi_hop(ppr_scores)
        
        # Make the RPE symmetric to (a,b) and (b,a)
        rpe_ba = rpe_ab.clone()  # In a real implementation this would use another forward pass
        
        return rpe_ab + rpe_ba

class MultiHeadAttention(nn.Module):
    def __init__(self, in_features, hidden_dim, rpe_dim, n_heads=4):
        """
        Multi-head attention layer for LPFormer.
        
        Args:
            in_features: Dimension of input features.
            hidden_dim: Dimension of hidden features.
            rpe_dim: Dimension of relative positional encoding.
            n_heads: Number of attention heads.
        """
        super(MultiHeadAttention, self).__init__()
        
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.rpe_dim = rpe_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        
        # Create attention heads
        self.attention_heads = nn.ModuleList([
            GATv2Attention(in_features, self.head_dim) 
            for _ in range(n_heads)
        ])
        
        # Projection for node features
        self.W_h = nn.Linear(in_features, hidden_dim, bias=False)
        
        # Output projection
        self.W_o = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, h_a, h_b, nodes_features, rpes, node_mask=None):
        """
        Compute multi-head attention.
        
        Args:
            h_a: Features of node a in the target link.
            h_b: Features of node b in the target link.
            nodes_features: Features of nodes being attended to.
            rpes: Relative positional encodings.
            node_mask: Mask for valid nodes.
            
        Returns:
            Context vector after attention.
        """
        batch_size = h_a.size(0)
        n_nodes = nodes_features.size(1) if nodes_features.dim() > 2 else 1
        
        # Reshape for batch processing if needed
        if nodes_features.dim() == 2:
            nodes_features = nodes_features.unsqueeze(0).expand(batch_size, -1, -1)
        if rpes.dim() == 2:
            rpes = rpes.unsqueeze(0).expand(batch_size, -1, -1)
        
        h_a = h_a.unsqueeze(1).expand(-1, n_nodes, -1)
        h_b = h_b.unsqueeze(1).expand(-1, n_nodes, -1)
        
        # Initialize attention weights and values
        all_attentions = []
        all_values = []
        
        # Calculate attention for each head
        for head in self.attention_heads:
            # Compute raw attention scores
            attn_weights = head(h_a, h_b, nodes_features, rpes)  # [batch_size, n_nodes, 1]
            
            # Apply mask if provided
            if node_mask is not None:
                attn_weights = attn_weights.masked_fill(~node_mask.unsqueeze(-1), -1e9)
            
            # Apply softmax to get attention weights
            attn_weights = F.softmax(attn_weights, dim=1)
            
            # Project node features
            values = self.W_h(nodes_features)  # [batch_size, n_nodes, hidden_dim]
            
            # Reshape for multi-head attention
            values = values.view(batch_size, n_nodes, self.n_heads, self.head_dim)
            values = values.transpose(1, 2)  # [batch_size, n_heads, n_nodes, head_dim]
            
            # Apply attention weights
            head_output = torch.matmul(attn_weights.transpose(1, 2), values)  # [batch_size, n_heads, 1, head_dim]
            
            all_attentions.append(attn_weights)
            all_values.append(head_output)
        
        # Concatenate outputs from all heads
        context = torch.cat(all_values, dim=2)  # [batch_size, 1, hidden_dim]
        context = context.squeeze(1)  # [batch_size, hidden_dim]
        
        # Final projection
        output = self.W_o(context)
        
        return output

class LPFormer(nn.Module):
    def __init__(self, in_features, hidden_dim, rpe_dim, n_heads=4, n_layers=2, dropout=0.1):
        """
        LPFormer model for link prediction.
        
        Args:
            in_features: Number of input features.
            hidden_dim: Number of hidden features.
            rpe_dim: Dimension of relative positional encoding.
            n_heads: Number of attention heads.
            n_layers: Number of transformer layers.
            dropout: Dropout rate.
        """
        super(LPFormer, self).__init__()
        
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.rpe_dim = rpe_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        
        # GCN for initial node embedding
        self.gcn = GCNConv(in_features, hidden_dim)
        
        # Relative positional encoding module
        self.rpe = RelativePositionalEncoding(2, rpe_dim)  # 2 PPR scores for (a,u) and (b,u)
        
        # Multi-head attention layers
        self.attention_layers = nn.ModuleList([
            MultiHeadAttention(hidden_dim, hidden_dim, rpe_dim, n_heads)
            for _ in range(n_layers)
        ])
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(n_layers)
        ])
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Final MLP for prediction
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + hidden_dim + 3, hidden_dim),  # node features + pairwise encoding + 3 counts
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
    
    def message_passing(self, x, edge_index):
        """
        Perform message passing using GCN.
        """
        return self.gcn(x, edge_index)
    
    def get_node_types(self, a_idx, b_idx, ppr_data, node_indices):
        """
        Determine the type of each node (CN, 1-hop, >1-hop) based on PPR scores.
        """
        ppr_matrix = ppr_data['ppr_matrix']
        cn_threshold = ppr_data['cn_threshold']
        one_hop_threshold = ppr_data['one_hop_threshold']
        multi_hop_threshold = ppr_data['multi_hop_threshold']
        
        # Get PPR scores
        batch_size = a_idx.size(0)
        device = a_idx.device
        
        cn_counts = torch.zeros(batch_size, device=device)
        one_hop_counts = torch.zeros(batch_size, device=device)
        multi_hop_counts = torch.zeros(batch_size, device=device)
        
        node_types = []
        filtered_nodes = []
        
        for i in range(batch_size):
            a = a_idx[i].item()
            b = b_idx[i].item()
            
            batch_nodes = []
            batch_types = []
            
            # Process each node
            for node_idx in node_indices:
                if node_idx == a or node_idx == b:
                    continue
                
                # Get PPR scores
                ppr_a_u = ppr_matrix[a, node_idx]
                ppr_b_u = ppr_matrix[b, node_idx]
                
                # Determine node type based on PPR scores
                if ppr_a_u > cn_threshold and ppr_b_u > cn_threshold:
                    node_type = 'cn'
                    cn_counts[i] += 1
                    batch_nodes.append(node_idx)
                    batch_types.append(node_type)
                elif ppr_a_u > one_hop_threshold or ppr_b_u > one_hop_threshold:
                    node_type = 'one_hop'
                    one_hop_counts[i] += 1
                    batch_nodes.append(node_idx)
                    batch_types.append(node_type)
                elif ppr_a_u > multi_hop_threshold or ppr_b_u > multi_hop_threshold:
                    node_type = 'multi_hop'
                    multi_hop_counts[i] += 1
                    batch_nodes.append(node_idx)
                    batch_types.append(node_type)
            
            filtered_nodes.append(batch_nodes)
            node_types.append(batch_types)
        
        counts = torch.stack([cn_counts, one_hop_counts, multi_hop_counts], dim=1)
        
        return filtered_nodes, node_types, counts
    
    def compute_rpe(self, a_idx, b_idx, ppr_data, node_indices, node_types):
        """
        Compute relative positional encoding based on PPR scores.
        """
        ppr_matrix = ppr_data['ppr_matrix']
        batch_size = a_idx.size(0)
        device = a_idx.device
        
        all_rpes = []
        
        for i in range(batch_size):
            a = a_idx[i].item()
            b = b_idx[i].item()
            
            batch_rpes = []
            
            for j, node_idx in enumerate(node_indices[i]):
                # Get PPR scores
                ppr_a_u = ppr_matrix[a, node_idx]
                ppr_b_u = ppr_matrix[b, node_idx]
                
                # Create PPR score tensor
                ppr_scores = torch.tensor([ppr_a_u, ppr_b_u], device=device)
                
                # Compute RPE
                node_type = node_types[i][j]
                rpe = self.rpe(ppr_scores.unsqueeze(0), node_type).squeeze(0)
                
                batch_rpes.append(rpe)
            
            # If there are no nodes for this batch item, add a zero tensor
            if not batch_rpes:
                batch_rpes.append(torch.zeros(self.rpe_dim, device=device))
            
            # Stack RPEs for this batch item
            all_rpes.append(torch.stack(batch_rpes, dim=0))
        
        return all_rpes
    
    def forward(self, x, edge_index, a_idx, b_idx, ppr_data):
        """
        Forward pass of LPFormer.
        
        Args:
            x: Node features.
            edge_index: Edge index.
            a_idx: Indices of source nodes.
            b_idx: Indices of target nodes.
            ppr_data: PPR matrix and thresholds.
            
        Returns:
            Probability of link existence.
        """
        # Message passing to get node embeddings
        h = self.message_passing(x, edge_index)
        
        batch_size = a_idx.size(0)
        device = a_idx.device
        
        # Get node representations for source and target nodes
        h_a = h[a_idx]  # [batch_size, hidden_dim]
        h_b = h[b_idx]  # [batch_size, hidden_dim]
        
        # Element-wise product of node features
        h_elem = h_a * h_b  # [batch_size, hidden_dim]
        
        # Get all node indices
        all_node_indices = list(range(h.size(0)))
        
        # Determine node types and filter nodes based on PPR scores
        filtered_nodes, node_types, counts = self.get_node_types(a_idx, b_idx, ppr_data, all_node_indices)
        
        # Compute relative positional encodings
        all_rpes = self.compute_rpe(a_idx, b_idx, ppr_data, filtered_nodes, node_types)
        
        # Initialize pairwise encodings
        pairwise_encodings = torch.zeros(batch_size, self.hidden_dim, device=device)
        
        # Process each batch item
        for i in range(batch_size):
            # Get node features and RPEs for this batch item
            if not filtered_nodes[i]:
                continue
                
            nodes_features = h[filtered_nodes[i]]  # [n_nodes, hidden_dim]
            rpes = all_rpes[i]  # [n_nodes, rpe_dim]
            
            # Apply attention layers
            x = nodes_features
            
            for j, (attn_layer, norm_layer) in enumerate(zip(self.attention_layers, self.layer_norms)):
                # Multi-head attention
                attn_output = attn_layer(h_a[i:i+1], h_b[i:i+1], x, rpes)
                
                # Residual connection and layer normalization
                x = norm_layer(attn_output + (x.mean(dim=0) if x.size(0) > 0 else torch.zeros_like(attn_output)))
                
                # Apply dropout
                x = self.dropout(x)
            
            # Store pairwise encoding
            pairwise_encodings[i] = x
        
        # Concatenate features for final prediction
        final_features = torch.cat([h_elem, pairwise_encodings, counts], dim=1)
        
        # Apply final MLP
        scores = self.mlp(final_features)
        
        return torch.sigmoid(scores.squeeze(1))

#####################################################################
# TRAINING AND EVALUATION
#####################################################################

def train_lpformer(model, optimizer, edge_data, split_data, ppr_data, num_epochs=100, patience=10):
    """
    Train the LPFormer model.
    """
    # Set model to training mode
    model.train()
    
    # Get data
    node_features = edge_data['node_features'].to(device)
    train_edge_index = split_data['train_edge_index'].to(device)
    val_pos_edge_index = split_data['val_pos_edge_index'].to(device)
    val_neg_edge_index = split_data['val_neg_edge_index'].to(device)
    
    # Create positive and negative samples for training
    pos_a_idx = train_edge_index[0, ::2]  # Take every other edge to avoid duplicates
    pos_b_idx = train_edge_index[1, ::2]
    
    # Create labels
    num_pos = pos_a_idx.size(0)
    pos_y = torch.ones(num_pos, device=device)
    
    # Set up early stopping
    best_val_auc = 0.0
    no_improve = 0
    
    # Training loop
    for epoch in range(num_epochs):
        print("\nEpoch no.: {epoch}")
        # Sample random negative edges for training
        neg_a_idx, neg_b_idx = sample_negative_edges(
            train_edge_index, edge_data['num_nodes'], num_pos)
        neg_y = torch.zeros(num_pos, device=device)
        
        # Combine positive and negative samples
        a_idx = torch.cat([pos_a_idx, neg_a_idx])
        b_idx = torch.cat([pos_b_idx, neg_b_idx])
        y = torch.cat([pos_y, neg_y])
        
        # Shuffle
        perm = torch.randperm(a_idx.size(0))
        a_idx = a_idx[perm]
        b_idx = b_idx[perm]
        y = y[perm]
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        pred = model(node_features, train_edge_index, a_idx, b_idx, ppr_data)
        
        # Compute loss
        loss = F.binary_cross_entropy(pred, y)
        
        # Backward pass
        loss.backward()
        
        # Update parameters
        optimizer.step()
        
        # Evaluate on validation set
        val_auc = evaluate(model, node_features, train_edge_index, 
                          val_pos_edge_index, val_neg_edge_index, ppr_data)
        
        # Print progress
        if (epoch + 1) % 5 == 0:
            print(f'Epoch: {epoch+1:02d}, Loss: {loss.item():.4f}, Val AUC: {val_auc:.4f}')
        
        # Check for early stopping
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            no_improve = 0
            # Save best model
            best_model_state = model.state_dict()
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    return model

def sample_negative_edges(edge_index, num_nodes, num_samples):
    """
    Sample random negative edges not in the graph.
    """
    # Convert edge index to set for O(1) lookup
    edge_set = set()
    for i in range(edge_index.size(1)):
        edge_set.add((edge_index[0, i].item(), edge_index[1, i].item()))
    
    # Sample negative edges
    neg_a_idx = []
    neg_b_idx = []
    
    while len(neg_a_idx) < num_samples:
        i = torch.randint(0, num_nodes, (1,)).item()
        j = torch.randint(0, num_nodes, (1,)).item()
        
        if i != j and (i, j) not in edge_set and (j, i) not in edge_set:
            neg_a_idx.append(i)
            neg_b_idx.append(j)
    
    return torch.tensor(neg_a_idx, device=edge_index.device), torch.tensor(neg_b_idx, device=edge_index.device)

def evaluate(model, node_features, train_edge_index, pos_edge_index, neg_edge_index, ppr_data):
    """
    Evaluate the model on the validation or test set.
    """
    # Set model to evaluation mode
    model.eval()
    
    with torch.no_grad():
        # Positive edges
        pos_a_idx = pos_edge_index[0]
        pos_b_idx = pos_edge_index[1]
        pos_y = torch.ones(pos_a_idx.size(0), device=pos_a_idx.device)
        
        # Negative edges
        neg_a_idx = neg_edge_index[0]
        neg_b_idx = neg_edge_index[1]
        neg_y = torch.zeros(neg_a_idx.size(0), device=neg_a_idx.device)
        
        # Combine positive and negative samples
        a_idx = torch.cat([pos_a_idx, neg_a_idx])
        b_idx = torch.cat([pos_b_idx, neg_b_idx])
        y = torch.cat([pos_y, neg_y])
        
        # Forward pass
        pred = model(node_features, train_edge_index, a_idx, b_idx, ppr_data)
        
        # Compute AUC
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y.cpu().numpy(), pred.cpu().numpy())
    
    return auc

def predict_links(model, node_features, edge_index, a_idx, b_idx, ppr_data):
    """
    Predict links for given node pairs.
    """
    # Set model to evaluation mode
    model.eval()
    
    with torch.no_grad():
        # Forward pass
        pred = model(node_features, edge_index, a_idx, b_idx, ppr_data)
    
    return pred

def get_top_k_predictions(model, node_features, edge_index, node_mapping, reverse_mapping, ppr_data, k=10):
    """
    Get top-k link predictions for nodes not currently connected.
    """
    # Set model to evaluation mode
    model.eval()
    
    num_nodes = node_features.size(0)
    device = node_features.device
    
    # Convert edge index to set for O(1) lookup
    edge_set = set()
    for i in range(edge_index.size(1)):
        edge_set.add((edge_index[0, i].item(), edge_index[1, i].item()))
    
    # Create all possible node pairs
    all_pairs = []
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):  # Only upper triangular to avoid duplicates
            if (i, j) not in edge_set and (j, i) not in edge_set:
                all_pairs.append((i, j))
    
    # Predict for all pairs in batches
    batch_size = 1024
    all_scores = []
    
    for i in range(0, len(all_pairs), batch_size):
        batch_pairs = all_pairs[i:i+batch_size]
        a_idx = torch.tensor([p[0] for p in batch_pairs], device=device)
        b_idx = torch.tensor([p[1] for p in batch_pairs], device=device)
        
        # Forward pass
        with torch.no_grad():
            scores = model(node_features, edge_index, a_idx, b_idx, ppr_data)
        
        all_scores.extend(scores.cpu().numpy())
    
    # Get top-k predictions
    pair_scores = list(zip(all_pairs, all_scores))
    pair_scores.sort(key=lambda x: x[1], reverse=True)
    
    top_k = pair_scores[:k]
    
    # Convert indices to original node names
    top_k_pairs = [(reverse_mapping[i], reverse_mapping[j], score) for (i, j), score in top_k]
    
    return top_k_pairs

#####################################################################
# MAIN FUNCTION
#####################################################################

def main():
    """
    Main function to run the LPFormer model.
    """
    # Load and preprocess data
    print("Loading data...")
    nodes_df, edges_df, hero_edges_df = load_marvel_data()
    
    print("Preprocessing data...")
    edge_data, G = preprocess_data(nodes_df, edges_df, hero_edges_df)
    
    print("Splitting edges...")
    split_data = split_edges(edge_data)
    
    # Compute PPR matrix
    print("Computing PPR matrix...")
    ppr_matrix = compute_ppr_matrix(
        edge_data['edge_index'], 
        edge_data['num_nodes'],
        alpha=0.15,
        eps=1e-5
    )
    
    # Apply threshold
    ppr_data = threshold_ppr_matrix(
        ppr_matrix,
        cn_threshold=1e-2,
        one_hop_threshold=1e-4,
        multi_hop_threshold=1e-3
    )
    
    # Define model parameters
    in_features = edge_data['node_features'].size(1)
    hidden_dim = 64
    rpe_dim = 32
    n_heads = 4
    n_layers = 2
    dropout = 0.1
    
    # Initialize model
    model = LPFormer(
        in_features=in_features,
        hidden_dim=hidden_dim,
        rpe_dim=rpe_dim,
        n_heads=n_heads,
        n_layers=n_layers,
        dropout=dropout
    ).to(device)
    
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # Train model
    print("Training model...")
    model = train_lpformer(
        model, 
        optimizer, 
        edge_data, 
        split_data, 
        ppr_data,
        num_epochs=100,
        patience=10
    )
    
    # Evaluate on test set
    print("Evaluating model...")
    test_auc = evaluate(
        model,
        edge_data['node_features'].to(device),
        split_data['train_edge_index'].to(device),
        split_data['test_pos_edge_index'].to(device),
        split_data['test_neg_edge_index'].to(device),
        ppr_data
    )
    
    print(f"Test AUC: {test_auc:.4f}")
    
    # Get top-k predictions
    print("Getting top predictions...")
    top_predictions = get_top_k_predictions(
        model,
        edge_data['node_features'].to(device),
        split_data['train_edge_index'].to(device),
        edge_data['node_mapping'],
        edge_data['reverse_mapping'],
        ppr_data,
        k=10
    )
    
    print("Top 10 predicted links:")
    for hero1, hero2, score in top_predictions:
        print(f"{hero1} -- {hero2}: {score:.4f}")
    
    # Visualization
    print("Creating visualization...")
    visualize_graph(G, top_predictions[:5])

def visualize_graph(G, top_predictions):
    """
    Visualize the network with top predicted links.
    """
    plt.figure(figsize=(12, 10))
    
    # Create a subgraph with nodes in top predictions
    nodes = set()
    for hero1, hero2, _ in top_predictions:
        nodes.add(hero1)
        nodes.add(hero2)
    
    # Add neighbors of these nodes
    for node in list(nodes):
        if node in G:
            nodes.update(G.neighbors(node))
    
    # Create subgraph
    subgraph = G.subgraph(nodes)
    
    # Define positions
    pos = nx.spring_layout(subgraph, seed=42)
    
    # Draw existing edges
    nx.draw_networkx_edges(subgraph, pos, alpha=0.3)
    
    # Draw predicted edges
    pred_edges = [(hero1, hero2) for hero1, hero2, _ in top_predictions]
    nx.draw_networkx_edges(subgraph, pos, edgelist=pred_edges, 
                          edge_color='r', width=2, alpha=0.8)
    
    # Draw nodes
    nx.draw_networkx_nodes(subgraph, pos)
    
    # Draw labels
    nx.draw_networkx_labels(subgraph, pos, font_size=8)
    
    plt.title("Hero Network with Top Predicted Links")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("hero_network_predictions.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    main()