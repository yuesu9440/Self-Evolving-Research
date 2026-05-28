#!/usr/bin/env python3
\"\"\"Cluster analysis for hydrogel network\"\"\"
import numpy as np
import networkx as nx

class ClusterAnalyzer:
    def __init__(self, cutoff=4.5):
        self.cutoff = cutoff
    
    def analyze_frame(self, positions):
        G = nx.Graph()
        for i, p in enumerate(positions):
            G.add_node(i, pos=p)
        for i in range(len(positions)):
            for j in range(i+1, len(positions)):
                d = np.linalg.norm(positions[i] - positions[j])
                if d < self.cutoff:
                    G.add_edge(i, j, distance=d)
        
        clusters = list(nx.connected_components(G))
        sizes = [len(c) for c in clusters]
        return {
            \"n_clusters\": len(clusters),
            \"max_cluster\": max(sizes) if sizes else 0,
            \"avg_cluster\": np.mean(sizes) if sizes else 0
        }
    
    def analyze_trajectory(self, frames):
        results = []
        for f in frames:
            results.append(self.analyze_frame(f))
        return results

if __name__ == \"__main__\":
    ca = ClusterAnalyzer()
    test_pos = np.random.rand(100, 3) * 50
    r = ca.analyze_frame(test_pos)
    print(f\"Clusters: {r['n_clusters']}, Max size: {r['max_cluster']}\")
