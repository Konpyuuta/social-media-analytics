#!/usr/bin/env python
# coding: utf-8

"""
Run LPFormer for link prediction on the Marvel Hero Network.
This script demonstrates how to use LPFormer for link prediction tasks.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from lpformer import (
    load_marvel_data,
    preprocess_data,
    split_edges,
    compute_ppr_matrix,
    threshold_ppr_matrix,
    LPFormer,
    train_lpformer,
    evaluate,
    get_top_k_predictions,
    visualize_graph
)

def run():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load and preprocess data
    print("Loading data...")
    nodes_df, edges_df, hero_edges_df = load_marvel_data()
    
    print("Preprocessing data...")
    edge_data, G = preprocess_data(nodes_df, edges_df, hero_edges_df)
    
    print("Splitting edges...")
    split_data = split_edges(edge_data)
    
    # Compute PPR matrix (can be slow for large graphs)
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
    hidden_dim = 64  # Dimensionality of hidden layers
    rpe_dim = 32     # Dimensionality of relative positional encoding
    n_heads = 4      # Number of attention heads
    n_layers = 2     # Number of transformer layers
    dropout = 0.1    # Dropout rate
    
    # Initialize model
    print(f"Building LPFormer model with {in_features} input features, "
          f"{hidden_dim} hidden dimensions, {n_heads} attention heads, "
          f"and {n_layers} layers...")
    
    model = LPFormer(
        in_features=in_features,
        hidden_dim=hidden_dim,
        rpe_dim=rpe_dim,
        n_heads=n_heads,
        n_layers=n_layers,
        dropout=dropout
    ).to(device)
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # Train model
    print("Training model...")
    model = train_lpformer(
        model, 
        optimizer, 
        edge_data, 
        split_data, 
        ppr_data,
        num_epochs=50,  # Reduced for demonstration
        patience=10
    )
    
    # Evaluate on test set
    print("Evaluating model on test set...")
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
        k=20  # Number of top predictions to return
    )
    
    print("\nTop 20 predicted links:")
    print("-" * 50)
    print(f"{'Hero 1':<20} | {'Hero 2':<20} | Score")
    print("-" * 50)
    for hero1, hero2, score in top_predictions:
        print(f"{hero1:<20} | {hero2:<20} | {score:.4f}")
    
    # Save predictions to file
    with open("top_hero_predictions.txt", "w") as f:
        f.write("Hero 1,Hero 2,Score\n")
        for hero1, hero2, score in top_predictions:
            f.write(f"{hero1},{hero2},{score:.4f}\n")
    
    print("\nPredictions saved to 'top_hero_predictions.txt'")
    
    # Visualization
    print("\nCreating visualization of top 5 predictions...")
    visualize_graph(G, top_predictions[:5])
    print("Visualization saved to 'hero_network_predictions.png'")
    
    return model, edge_data, split_data, ppr_data, top_predictions

if __name__ == "__main__":
    run()