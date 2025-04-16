import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
from collections import Counter, defaultdict
from lpformer import load_marvel_data, preprocess_data

def analyze_predictions(predictions_file="top_hero_predictions.txt"):
    """
    Analyze the predicted links and their patterns.
    """
    # Load predictions
    if not os.path.exists(predictions_file):
        print(f"Predictions file {predictions_file} not found.")
        return
    
    predictions_df = pd.read_csv(predictions_file)
    print(f"Loaded {len(predictions_df)} predictions.")
    
    # Load original data
    nodes_df, edges_df, hero_edges_df = load_marvel_data()
    edge_data, G = preprocess_data(nodes_df, edges_df, hero_edges_df)
    
    # Analyze common neighbors for predicted links
    print("\nAnalyzing common neighbor patterns...")
    cn_counts = []
    for _, row in predictions_df.iterrows():
        hero1, hero2 = row['Hero 1'], row['Hero 2']
        if hero1 in G and hero2 in G:
            cn = list(nx.common_neighbors(G, hero1, hero2))
            cn_counts.append(len(cn))
        else:
            cn_counts.append(0)
    
    predictions_df['CN_Count'] = cn_counts
    
    # Analyze comics in common
    print("\nAnalyzing comics in common...")
    comics_in_common = []
    hero_to_comics = defaultdict(set)
    
    # Create hero-to-comics mapping
    for _, row in edges_df.iterrows():
        hero_to_comics[row['hero']].add(row['comic'])
    
    # Count comics in common for predicted links
    for _, row in predictions_df.iterrows():
        hero1, hero2 = row['Hero 1'], row['Hero 2']
        common_comics = hero_to_comics[hero1].intersection(hero_to_comics[hero2])
        comics_in_common.append(len(common_comics))
    
    predictions_df['Comics_In_Common'] = comics_in_common
    
    # Analyze node degrees
    print("\nAnalyzing node degree patterns...")
    hero1_degrees = []
    hero2_degrees = []
    
    for _, row in predictions_df.iterrows():
        hero1, hero2 = row['Hero 1'], row['Hero 2']
        hero1_degree = G.degree(hero1) if hero1 in G else 0
        hero2_degree = G.degree(hero2) if hero2 in G else 0
        hero1_degrees.append(hero1_degree)
        hero2_degrees.append(hero2_degree)
    
    predictions_df['Hero1_Degree'] = hero1_degrees
    predictions_df['Hero2_Degree'] = hero2_degrees
    predictions_df['Avg_Degree'] = (predictions_df['Hero1_Degree'] + predictions_df['Hero2_Degree']) / 2
    
    # Save enhanced predictions
    predictions_df.to_csv("enhanced_predictions.csv", index=False)
    print("Enhanced predictions saved to 'enhanced_predictions.csv'")
    
    # Visualize relationships
    print("\nCreating visualizations...")
    
    # Create a figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Score vs Common Neighbors
    axes[0, 0].scatter(predictions_df['CN_Count'], predictions_df['Score'])
    axes[0, 0].set_xlabel('Number of Common Neighbors')
    axes[0, 0].set_ylabel('Prediction Score')
    axes[0, 0].set_title('Score vs Common Neighbors')
    
    # Plot 2: Score vs Comics in Common
    axes[0, 1].scatter(predictions_df['Comics_In_Common'], predictions_df['Score'])
    axes[0, 1].set_xlabel('Number of Comics in Common')
    axes[0, 1].set_ylabel('Prediction Score')
    axes[0, 1].set_title('Score vs Comics in Common')
    
    # Plot 3: Score vs Average Degree
    axes[1, 0].scatter(predictions_df['Avg_Degree'], predictions_df['Score'])
    axes[1, 0].set_xlabel('Average Degree')
    axes[1, 0].set_ylabel('Prediction Score')
    axes[1, 0].set_title('Score vs Average Degree')
    
    # Plot 4: Comics in Common vs Common Neighbors with Score as color
    scatter = axes[1, 1].scatter(
        predictions_df['Comics_In_Common'], 
        predictions_df['CN_Count'],
        c=predictions_df['Score'],
        cmap='viridis',
        alpha=0.7
    )
    axes[1, 1].set_xlabel('Comics in Common')
    axes[1, 1].set_ylabel('Common Neighbors')
    axes[1, 1].set_title('Comics in Common vs Common Neighbors (colored by Score)')
    plt.colorbar(scatter, ax=axes[1, 1], label='Score')
    
    plt.tight_layout()
    plt.savefig('prediction_analysis.png', dpi=300)
    plt.close()
    
    # Calculate correlations
    print("\nCalculating correlations between features and prediction scores:")
    correlation_df = predictions_df[['Score', 'CN_Count', 'Comics_In_Common', 'Hero1_Degree', 'Hero2_Degree', 'Avg_Degree']]
    correlation_matrix = correlation_df.corr()
    
    print(correlation_matrix['Score'].sort_values(ascending=False))
    
    # Visualize correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png', dpi=300)
    plt.close()
    
    print("Visualizations saved as 'prediction_analysis.png' and 'correlation_matrix.png'")
    
    # Find the most popular heroes in predictions
    hero1_counts = Counter(predictions_df['Hero 1'])
    hero2_counts = Counter(predictions_df['Hero 2'])
    hero_counts = hero1_counts + hero2_counts
    
    print("\nMost frequent heroes in predictions:")
    for hero, count in hero_counts.most_common(10):
        print(f"{hero}: {count} occurrences")
    
    # Analyze network communities
    try:
        print("\nAnalyzing network communities...")
        communities = list(nx.community.greedy_modularity_communities(G))
        
        # Check if predicted links are within or across communities
        within_community = 0
        across_community = 0
        
        for _, row in predictions_df.iterrows():
            hero1, hero2 = row['Hero 1'], row['Hero 2']
            if hero1 not in G or hero2 not in G:
                continue
                
            # Find communities for both heroes
            hero1_community = None
            hero2_community = None
            
            for i, community in enumerate(communities):
                if hero1 in community:
                    hero1_community = i
                if hero2 in community:
                    hero2_community = i
            
            if hero1_community == hero2_community:
                within_community += 1
            else:
                across_community += 1
        
        total_links = within_community + across_community
        within_percent = (within_community / total_links) * 100 if total_links > 0 else 0
        across_percent = (across_community / total_links) * 100 if total_links > 0 else 0
        
        print(f"Within-community links: {within_community} ({within_percent:.1f}%)")
        print(f"Across-community links: {across_community} ({across_percent:.1f}%)")
        
        # Visualize community distribution
        plt.figure(figsize=(8, 6))
        plt.bar(['Within Community', 'Across Communities'], [within_community, across_community])
        plt.ylabel('Number of Predicted Links')
        plt.title('Distribution of Predicted Links Across Communities')
        plt.savefig('community_analysis.png', dpi=300)
        plt.close()
        
        print("Community analysis visualization saved as 'community_analysis.png'")
    except Exception as e:
        print(f"Error in community analysis: {e}")
    
    return predictions_df

if __name__ == "__main__":
    analyze_predictions()