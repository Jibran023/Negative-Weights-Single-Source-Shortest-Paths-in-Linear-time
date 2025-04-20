import math
import random
import heapq
import networkx as nx
from collections import defaultdict, deque
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np

# ---------------------------
# Negative Cycle Detection
# ---------------------------

def detect_negative_cycle(G):
    """
    Detect negative cycles in a graph using the Bellman-Ford algorithm
    Returns (has_cycle, cycle_nodes) where cycle_nodes is a list of nodes in a negative cycle if found
    """
    # Choose arbitrary source node
    if not G.nodes:
        return False, []
        
    source = list(G.nodes)[0]
    n = len(G.nodes)
    
    # Initialize distances and predecessors
    dist = {node: float('inf') for node in G.nodes}
    dist[source] = 0
    pred = {node: None for node in G.nodes}
    
    # Relax edges repeatedly
    for _ in range(n):
        relaxed = False
        for u, v, data in G.edges(data=True):
            weight = data['weight']
            if dist[u] != float('inf') and dist[u] + weight < dist[v]:
                dist[v] = dist[u] + weight
                pred[v] = u
                relaxed = True
        
        if not relaxed:
            return False, []  # No negative cycle
    
    # Check for negative cycle
    for u, v, data in G.edges(data=True):
        weight = data['weight']
        if dist[u] != float('inf') and dist[u] + weight < dist[v]:
            # Negative cycle exists, reconstruct it
            cycle_node = v
            for _ in range(n):
                cycle_node = pred[cycle_node]  # Follow predecessors to find a node in the cycle
            
            # Extract the cycle
            cycle = [cycle_node]
            current = pred[cycle_node]
            while current != cycle_node:
                cycle.append(current)
                current = pred[current]
            cycle.append(cycle_node)  # Close the cycle
            
            return True, cycle[::-1]  # Return reversed to get correct order
            
    return False, []  # No negative cycle

# ---------------------------
# Shortest Path Finding
# ---------------------------

def find_shortest_paths(G, source):
    """
    Finds shortest paths in a graph that may contain negative edges
    
    Args:
        G: NetworkX DiGraph with weighted edges
        source: Source node for shortest paths
        
    Returns:
        (distances, predecessors) or (None, None) if negative cycle detected
    """
    # First, check for negative cycles
    has_cycle, cycle = detect_negative_cycle(G)
    
    if has_cycle:
        print(f"Negative cycle detected: {' -> '.join(str(n) for n in cycle)}")
        print("The graph contains a negative cycle, so shortest paths are not well-defined.")
        
        # Calculate the cycle weight to show the user
        cycle_weight = sum(G[cycle[i]][cycle[i+1]]['weight'] for i in range(len(cycle)-1))
        print(f"Total cycle weight: {cycle_weight}")
        
        return None, None
    
    # No negative cycles, use Bellman-Ford algorithm
    try:
        # Try using NetworkX's implementation
        predecessors, distances = nx.bellman_ford_predecessor_and_distance(G, source)
        return distances, predecessors
    except Exception as e:
        print(f"Error computing shortest paths: {e}")
        return None, None

# ---------------------------
# Graph Visualization
# ---------------------------

def visualize_graphs(G, source=None, distances=None, predecessors=None, modified_G=None, title=None):
    """
    Visualize original graph and shortest paths
    
    Args:
        G: Original NetworkX DiGraph
        source: Source node for shortest paths
        distances: Dictionary of shortest path distances from source
        predecessors: Dictionary of predecessors in shortest paths
        modified_G: Modified graph (optional)
        title: Title for the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    if title:
        fig.suptitle(title, fontsize=16)
    
    # Position nodes (use same layout for both graphs)
    pos = nx.spring_layout(G, seed=42)
    
    # Plot original graph
    ax = axes[0]
    ax.set_title("Original Graph")
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color='lightblue', node_size=500)
    
    # Draw edges with weights
    edge_labels = {(u, v): f"{d['weight']}" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edges(G, pos, ax=ax, arrowsize=15)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax)
    
    # Highlight negative edges in red
    neg_edges = [(u, v) for u, v, d in G.edges(data=True) if d['weight'] < 0]
    nx.draw_networkx_edges(G, pos, edgelist=neg_edges, edge_color='red', ax=ax, arrowsize=15)
    
    # Draw node labels
    nx.draw_networkx_labels(G, pos, ax=ax)
    
    # Plot graph with shortest paths
    ax = axes[1]
    
    if distances is not None and predecessors is not None and source is not None:
        # Create a subgraph showing shortest paths
        shortest_path_graph = nx.DiGraph()
        
        # Add nodes with distance information
        for node in G.nodes():
            if node in distances:
                label = f"{node}\ndist: {distances[node]}"
                shortest_path_graph.add_node(node, label=label)
        
        # Add edges in shortest paths
        for node in predecessors:
            for pred in predecessors[node]:
                if pred is not None:  # Skip source node which has no predecessor
                    shortest_path_graph.add_edge(pred, node, 
                                                weight=G[pred][node]['weight'])
        
        ax.set_title(f"Shortest Paths from {source}")
        
        # Draw nodes with distance information
        node_labels = {node: f"{node}\n{distances[node]}" for node in shortest_path_graph.nodes() if node in distances}
        nx.draw_networkx_nodes(shortest_path_graph, pos, ax=ax, node_color='lightgreen', node_size=600)
        nx.draw_networkx_labels(shortest_path_graph, pos, labels=node_labels, ax=ax, font_size=9)
        
        # Draw edges in shortest paths
        nx.draw_networkx_edges(shortest_path_graph, pos, ax=ax, arrowsize=15, edge_color='blue')
        
        # Show edge weights
        edge_labels = {(u, v): f"{G[u][v]['weight']}" for u, v in shortest_path_graph.edges()}
        nx.draw_networkx_edge_labels(shortest_path_graph, pos, edge_labels=edge_labels, ax=ax)
    
    elif modified_G is not None:
        # Draw the modified graph
        ax.set_title("Modified Graph")
        
        # Draw nodes
        nx.draw_networkx_nodes(modified_G, pos, ax=ax, node_color='lightgreen', node_size=500)
        
        # Draw edges with weights
        edge_labels = {(u, v): f"{d['weight']}" for u, v, d in modified_G.edges(data=True)}
        nx.draw_networkx_edges(modified_G, pos, ax=ax, arrowsize=15)
        nx.draw_networkx_edge_labels(modified_G, pos, edge_labels=edge_labels, ax=ax)
        
        # Highlight negative edges in red
        neg_edges = [(u, v) for u, v, d in modified_G.edges(data=True) if d['weight'] < 0]
        nx.draw_networkx_edges(modified_G, pos, edgelist=neg_edges, edge_color='red', ax=ax, arrowsize=15)
        
        # Draw node labels
        nx.draw_networkx_labels(modified_G, pos, ax=ax)
    
    else:
        ax.set_title("Cannot Display Shortest Paths - Negative Cycle Detected")
        ax.text(0.5, 0.5, "Shortest paths are not well-defined\ndue to negative cycle", 
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=12)
    
    # Remove axis ticks
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    plt.savefig('graph_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()

# ---------------------------
# Main Function
# ---------------------------

def main():
    # Create test graph with negative edges
    G = nx.DiGraph()
    G.add_weighted_edges_from([
        ('A', 'B', -4),
        ('A', 'C', 2),
        ('B', 'C', -3),
        ('B', 'D', 2),
        ('C', 'D', -4),
        ('C', 'E', 5),
        ('D', 'E', 5),
        ('E', 'B', -3)  # Creates a negative cycle
    ])
    
    print("Testing graph with negative edges and cycles:")
    
    # Check for negative cycles
    has_cycle, cycle = detect_negative_cycle(G)
    if has_cycle:
        print(f"Negative cycle detected: {' -> '.join(str(n) for n in cycle)}")
        
        # Calculate cycle weight
        cycle_weight = sum(G[cycle[i]][cycle[i+1]]['weight'] for i in range(len(cycle)-1))
        print(f"Total cycle weight: {cycle_weight}")
        
        # Visualize original graph with negative cycle
        visualize_graphs(G, title="Graph with Negative Cycle")
        
        # Modify graph to break negative cycle
        print("\nModifying graph to remove negative cycle:")
        G_modified = G.copy()
        G_modified.remove_edge('E', 'B')  # Remove the edge that completes the negative cycle
        
        # Visualize original and modified graphs
        visualize_graphs(G, modified_G=G_modified, title="Breaking the Negative Cycle")
        
        # Now try finding shortest paths in modified graph
        source = 'A'
        distances, predecessors = find_shortest_paths(G_modified, source)
        
        # Visualize shortest paths
        if distances:
            visualize_graphs(G_modified, source, distances, predecessors, 
                             title=f"Shortest Paths from {source} (After Breaking Cycle)")
    else:
        # Find shortest paths
        source = 'A'
        distances, predecessors = find_shortest_paths(G, source)
        
        # Visualize shortest paths
        if distances:
            visualize_graphs(G, source, distances, predecessors, 
                             title=f"Shortest Paths from {source}")
            
            print(f"\nShortest path distances from {source}:")
            for node in sorted(distances):
                print(f"{node}: {distances[node]}")

    print("\nGraph visualization saved as 'graph_visualization.png'")

if __name__ == "_main_":
    main()