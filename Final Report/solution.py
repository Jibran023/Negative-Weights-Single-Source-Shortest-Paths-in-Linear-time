import math
import random
import heapq
import networkx as nx
from collections import defaultdict, deque
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np
import time

# [Keep all existing functions from original code]
# ---------------------------
# Johnson's Algorithm
# ---------------------------

def johnson_reweight(G):
    """
    Apply Johnson's reweighting technique to transform a graph with negative edges
    to a graph with non-negative edges while preserving shortest paths
    
    Args:
        G: NetworkX DiGraph with weighted edges
        
    Returns:
        (reweighted_graph, potentials) or (None, None) if negative cycle detected
    """
    # Create a new graph with an extra node (we'll call it 'source')
    # connected to all nodes with zero-weight edges
    H = G.copy()
    source_node = 'temp_source'
    
    # Add source node with zero-weight edges to all nodes
    for node in G.nodes():
        H.add_edge(source_node, node, weight=0)
    
    # Run Bellman-Ford from the source node to get potentials
    try:
        # Try using NetworkX's implementation
        potentials = nx.bellman_ford_predecessor_and_distance(H, source_node)[1]
        
        # Create reweighted graph
        G_prime = G.copy()
        
        # Reweight edges using the potentials
        for u, v, data in G.edges(data=True):
            G_prime[u][v]['weight'] = data['weight'] + potentials[u] - potentials[v]
            # Store original weight for reference
            G_prime[u][v]['original_weight'] = data['weight']
        
        # Remove temporary source node from potentials
        del potentials[source_node]
        
        return G_prime, potentials
    
    except nx.NetworkXUnbounded:
        # Negative cycle detected
        print("Negative cycle detected. Johnson's algorithm cannot proceed.")
        return None, None

def convert_to_original_distance(distance, potentials, source, target):
    """
    Convert a distance in the reweighted graph back to the original graph
    
    Args:
        distance: Distance in reweighted graph
        potentials: Node potentials from Johnson's algorithm
        source: Source node
        target: Target node
    
    Returns:
        Original distance
    """
    return distance - potentials[source] + potentials[target]

def johnson_shortest_paths(G, source=None):
    """
    Find all-pairs shortest paths using Johnson's algorithm
    
    Args:
        G: NetworkX DiGraph with weighted edges
        source: Optional source node (if provided, only compute paths from this source)
        
    Returns:
        Dictionary of dictionaries with shortest path distances
    """
    # Apply Johnson's reweighting
    G_prime, potentials = johnson_reweight(G)
    
    if G_prime is None:
        return None  # Negative cycle detected
    
    # Use Dijkstra's algorithm on the reweighted graph
    if source is not None:
        # Single-source shortest paths
        sources = [source]
    else:
        # All-pairs shortest paths
        sources = G.nodes()
    
    # Dictionary to store shortest path distances
    distances = {}
    paths = {}
    
    # Run Dijkstra's algorithm from each source node
    for s in sources:
        # Get shortest paths in reweighted graph
        dijkstra_distances = nx.single_source_dijkstra(G_prime, s)
        
        # Convert back to original distances
        s_distances = {}
        s_paths = {}
        
        for target, path in dijkstra_distances[1].items():
            # Calculate original distance
            original_dist = dijkstra_distances[0][target]
            if original_dist != float('inf'):
                original_dist = convert_to_original_distance(original_dist, potentials, s, target)
            s_distances[target] = original_dist
            s_paths[target] = path
        
        distances[s] = s_distances
        paths[s] = s_paths
    
    return {'distances': distances, 'paths': paths, 'potentials': potentials, 'reweighted_graph': G_prime}

# ---------------------------
# Graph Decomposition
# ---------------------------

def decompose_into_sccs(G):
    """
    Decompose a graph into strongly connected components (SCCs)
    
    Args:
        G: NetworkX DiGraph
        
    Returns:
        List of subgraphs, each representing an SCC
    """
    # Find strongly connected components
    sccs = list(nx.strongly_connected_components(G))
    
    # Create subgraphs for each component
    scc_subgraphs = []
    for i, component in enumerate(sccs):
        # Create subgraph
        subgraph = G.subgraph(component).copy()
        # Store component id
        nx.set_node_attributes(subgraph, {node: i for node in subgraph.nodes()}, 'component_id')
        scc_subgraphs.append(subgraph)
    
    return scc_subgraphs

def create_condensation_graph(G, sccs):
    """
    Create a condensation graph where each node represents an SCC
    
    Args:
        G: Original NetworkX DiGraph
        sccs: List of sets of nodes, each representing an SCC
        
    Returns:
        Condensation graph (NetworkX DiGraph)
    """
    # Create mapping from node to SCC ID
    node_to_scc = {}
    for i, component in enumerate(sccs):
        for node in component:
            node_to_scc[node] = i
    
    # Create condensation graph
    C = nx.DiGraph()
    
    # Add nodes (one for each SCC)
    for i, component in enumerate(sccs):
        # Store the nodes in this component
        C.add_node(i, nodes=list(component))
    
    # Add edges between components
    for u, v, data in G.edges(data=True):
        scc_u = node_to_scc[u]
        scc_v = node_to_scc[v]
        
        # Only add edges between different components
        if scc_u != scc_v:
            # If edge already exists, choose the minimum weight
            if C.has_edge(scc_u, scc_v):
                C[scc_u][scc_v]['weight'] = min(C[scc_u][scc_v]['weight'], data['weight'])
            else:
                C.add_edge(scc_u, scc_v, weight=data['weight'])
    
    return C

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

def visualize_graph(G, pos, ax, title=None, node_colors=None, node_labels=None, 
                    highlight_edges=None, edge_colors=None, edge_labels=None):
    """
    Helper function to visualize a graph
    """
    if title:
        ax.set_title(title)
    
    # Default node colors
    if node_colors is None:
        node_colors = 'lightblue'
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=500)
    
    # Draw edges
    if edge_colors is None:
        nx.draw_networkx_edges(G, pos, ax=ax, arrowsize=15)
    else:
        # Draw edges with specific colors
        for color, edge_list in edge_colors.items():
            if edge_list:
                nx.draw_networkx_edges(G, pos, edgelist=edge_list, edge_color=color, ax=ax, arrowsize=15)
    
    # Highlight specific edges
    if highlight_edges:
        nx.draw_networkx_edges(G, pos, edgelist=highlight_edges, edge_color='red', ax=ax, arrowsize=15)
    
    # Draw edge labels
    if edge_labels:
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax)
    else:
        # Default edge labels are weights
        edge_labels = {(u, v): f"{d['weight']}" for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax)
    
    # Draw node labels
    if node_labels:
        nx.draw_networkx_labels(G, pos, labels=node_labels, ax=ax)
    else:
        nx.draw_networkx_labels(G, pos, ax=ax)
    
    # Remove axis ticks
    ax.set_xticks([])
    ax.set_yticks([])

def visualize_graphs(G, source=None, distances=None, predecessors=None, 
                     modified_G=None, title=None, reweighted_G=None, potentials=None):
    """
    Visualize original graph and shortest paths
    
    Args:
        G: Original NetworkX DiGraph
        source: Source node for shortest paths
        distances: Dictionary of shortest path distances from source
        predecessors: Dictionary of predecessors in shortest paths
        modified_G: Modified graph (optional)
        title: Title for the plot
        reweighted_G: Reweighted graph from Johnson's algorithm
        potentials: Node potentials from Johnson's algorithm
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    if title:
        fig.suptitle(title, fontsize=16)
    
    # Position nodes (use same layout for both graphs)
    pos = nx.spring_layout(G, seed=42)
    
    # Plot original graph
    ax = axes[0]
    
    # Determine negative edges
    neg_edges = [(u, v) for u, v, d in G.edges(data=True) if d['weight'] < 0]
    edge_colors = {'blue': [(u, v) for u, v, d in G.edges(data=True) if d['weight'] >= 0]}
    
    visualize_graph(G, pos, ax, title="Original Graph", 
                    highlight_edges=neg_edges,
                    edge_colors={'blue': [(u, v) for u, v in G.edges() if (u, v) not in neg_edges]})
    
    # Plot graph with shortest paths or reweighted graph
    ax = axes[1]
    
    if reweighted_G is not None and potentials is not None:
        # Show the reweighted graph from Johnson's algorithm
        ax.set_title("Reweighted Graph (Johnson's Algorithm)")
        
        # Node labels showing potentials
        node_labels = {node: f"{node}\nh({node})={potentials[node]}" for node in reweighted_G.nodes()}
        
        # Edge labels showing both original and reweighted values
        edge_labels = {(u, v): f"{d['weight']:.1f}\n(orig: {d['original_weight']})" 
                       for u, v, d in reweighted_G.edges(data=True)}
        
        # All edges should be non-negative now
        visualize_graph(reweighted_G, pos, ax, node_colors='lightgreen', 
                         node_labels=node_labels, edge_labels=edge_labels)
        
    elif distances is not None and predecessors is not None and source is not None:
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
        
        # Determine negative edges in shortest paths
        neg_sp_edges = [(u, v) for u, v in shortest_path_graph.edges() 
                         if G[u][v]['weight'] < 0]
        
        visualize_graph(shortest_path_graph, pos, ax, node_colors='lightgreen',
                        node_labels=node_labels, 
                        edge_colors={'blue': [(u, v) for u, v in shortest_path_graph.edges() 
                                              if (u, v) not in neg_sp_edges],
                                    'red': neg_sp_edges})
        
    elif modified_G is not None:
        # Draw the modified graph
        ax.set_title("Modified Graph")
        
        # Determine negative edges in modified graph
        neg_mod_edges = [(u, v) for u, v, d in modified_G.edges(data=True) if d['weight'] < 0]
        
        visualize_graph(modified_G, pos, ax, node_colors='lightgreen',
                        highlight_edges=neg_mod_edges)
    
    else:
        ax.set_title("Cannot Display Shortest Paths - Negative Cycle Detected")
        ax.text(0.5, 0.5, "Shortest paths are not well-defined\ndue to negative cycle", 
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=12)
    
    plt.tight_layout()
    plt.savefig('graph_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()

def visualize_components(G, components, title="Graph Decomposition into SCCs"):
    """
    Visualize graph decomposition into strongly connected components
    
    Args:
        G: Original NetworkX DiGraph
        components: List of subgraphs, each representing a component
        title: Title for the plot
    """
    # Determine number of components and arrange in grid
    n_comps = len(components)
    cols = min(3, n_comps)  # Max 3 columns
    rows = (n_comps + cols - 1) // cols  # Ceiling division
    
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
    fig.suptitle(title, fontsize=16)
    
    # Flatten axes array if it's a grid
    if rows > 1 and cols > 1:
        axes = axes.flatten()
    elif rows == 1 and cols > 1:
        axes = axes  # Already a 1D array
    elif rows > 1 and cols == 1:
        axes = axes.flatten()
    else:
        axes = [axes]  # Single subplot
    
    # Position nodes for entire graph
    pos = nx.spring_layout(G, seed=42)
    
    # Plot each component
    for i, component in enumerate(components):
        if i < len(axes):
            ax = axes[i]
            
            # Filter positions for this component
            comp_pos = {node: pos[node] for node in component.nodes()}
            
            # Get component ID
            comp_id = list(component.nodes())[0]
            if 'component_id' in component.nodes[comp_id]:
                comp_id = component.nodes[comp_id]['component_id']
            
            # Determine if component has negative cycle
            has_cycle, cycle = detect_negative_cycle(component)
            if has_cycle:
                title = f"Component {i} (Negative Cycle)"
                color = 'lightcoral'
            else:
                title = f"Component {i}"
                color = 'lightblue'
            
            # Draw component
            visualize_graph(component, comp_pos, ax, title=title, node_colors=color)
    
    # Hide unused subplots
    for i in range(n_comps, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for the title
    plt.savefig('component_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()

def visualize_condensation(G, C, components, title="Condensation Graph"):
    """
    Visualize the condensation graph
    
    Args:
        G: Original NetworkX DiGraph
        C: Condensation graph
        components: List of sets of nodes, each representing a component
        title: Title for the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle(title, fontsize=16)
    
    # Original graph with components colored
    pos_orig = nx.spring_layout(G, seed=42)
    
    # Create color map for nodes
    node_colors = []
    color_map = plt.cm.rainbow(np.linspace(0, 1, len(components)))
    
    # Assign colors to nodes based on component
    node_to_comp = {}
    for i, comp in enumerate(components):
        for node in comp:
            node_to_comp[node] = i
    
    for node in G.nodes():
        node_colors.append(color_map[node_to_comp[node]])
    
    # Draw original graph with colored components
    ax1.set_title("Original Graph with Colored Components")
    nx.draw_networkx(G, pos_orig, ax=ax1, node_color=node_colors, with_labels=True)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos_orig, ax=ax1, arrowsize=15)
    
    # Draw condensation graph
    ax2.set_title("Condensation Graph")
    
    # Position for condensation graph
    pos_cond = nx.spring_layout(C, seed=42)
    
    # Node labels showing component members
    node_labels = {i: f"C{i}\n{', '.join(str(n) for n in C.nodes[i]['nodes'])}" 
                   for i in C.nodes()}
    
    # Edge labels showing weights
    edge_labels = {(u, v): f"{d['weight']}" for u, v, d in C.edges(data=True)}
    
    # Draw condensation graph
    nx.draw_networkx_nodes(C, pos_cond, ax=ax2, node_color=color_map[:len(C)], 
                           node_size=700)
    nx.draw_networkx_labels(C, pos_cond, labels=node_labels, ax=ax2, font_size=9)
    nx.draw_networkx_edges(C, pos_cond, ax=ax2, arrowsize=15)
    nx.draw_networkx_edge_labels(C, pos_cond, edge_labels=edge_labels, ax=ax2)
    
    # Remove axis ticks
    for ax in (ax1, ax2):
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for the title
    plt.savefig('condensation_graph.png', dpi=300, bbox_inches='tight')
    plt.show()

# ---------------------------
# Performance Testing
# ---------------------------

def direct_dijkstra(G, source):
    """
    Apply Dijkstra's algorithm directly to a graph
    (Will fail if there are negative edges)
    
    Args:
        G: NetworkX DiGraph with weighted edges
        source: Source node
        
    Returns:
        Dictionary with shortest path distances
    """
    try:
        # Check if there are negative edges
        has_negative_edges = any(d['weight'] < 0 for _, _, d in G.edges(data=True))
        if has_negative_edges:
            print("Warning: Graph contains negative edges. Dijkstra's algorithm may give incorrect results.")
        
        # Run Dijkstra's algorithm
        return nx.single_source_dijkstra_path_length(G, source)
    except Exception as e:
        print(f"Error applying Dijkstra's algorithm: {e}")
        return None

def visualize_performance_comparison(nodes, johnson_times, bellman_ford_times, dijkstra_times=None):
    """
    Visualize performance comparison between algorithms
    
    Args:
        nodes: List of node counts
        johnson_times: List of execution times for Johnson's algorithm
        bellman_ford_times: List of execution times for Bellman-Ford algorithm
        dijkstra_times: List of execution times for Dijkstra's algorithm (optional)
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(nodes, johnson_times, 'o-', color='blue', label="Johnson's Algorithm")
    ax.plot(nodes, bellman_ford_times, 's-', color='red', label="Bellman-Ford Algorithm")
    
    if dijkstra_times:
        ax.plot(nodes, dijkstra_times, '^-', color='green', label="Dijkstra's Algorithm")
    
    ax.set_xlabel('Number of Nodes')
    ax.set_ylabel('Execution Time (seconds)')
    ax.set_title('Performance Comparison of Shortest Path Algorithms')
    ax.grid(True)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('algorithm_performance.png', dpi=300)
    plt.show()

def compare_algorithm_performance(G, source):
    """
    Compare the performance of Johnson's algorithm, Bellman-Ford, and Dijkstra
    for finding shortest paths
    
    Args:
        G: NetworkX DiGraph with weighted edges
        source: Source node
        
    Returns:
        Dictionary with execution times and results
    """
    results = {}
    
    # Check if graph has negative cycles
    has_cycle, cycle = detect_negative_cycle(G)
    if has_cycle:
        print("Graph contains negative cycles. Skipping performance comparison.")
        return None
    
    # Measure time for Johnson's algorithm
    print("\nTesting Johnson's algorithm performance...")
    start_time = time.time()
    johnson_result = johnson_shortest_paths(G, source)
    johnson_time = time.time() - start_time
    
    if johnson_result:
        results['johnson_time'] = johnson_time
        results['johnson_distances'] = johnson_result['distances'][source]
        print(f"Johnson's algorithm execution time: {johnson_time:.6f} seconds")
    
    # Measure time for Bellman-Ford algorithm
    print("\nTesting Bellman-Ford algorithm performance...")
    start_time = time.time()
    bf_distances, _ = find_shortest_paths(G, source)
    bf_time = time.time() - start_time
    
    if bf_distances:
        results['bellman_ford_time'] = bf_time
        results['bellman_ford_distances'] = bf_distances
        print(f"Bellman-Ford algorithm execution time: {bf_time:.6f} seconds")
    
    # Check if graph has negative edges
    has_negative_edges = any(d['weight'] < 0 for _, _, d in G.edges(data=True))
    
    # Measure time for Dijkstra's algorithm (only if no negative edges)
    if not has_negative_edges:
        print("\nTesting Dijkstra's algorithm performance...")
        start_time = time.time()
        dijkstra_distances = direct_dijkstra(G, source)
        dijkstra_time = time.time() - start_time
        
        if dijkstra_distances:
            results['dijkstra_time'] = dijkstra_time
            results['dijkstra_distances'] = dijkstra_distances
            print(f"Dijkstra's algorithm execution time: {dijkstra_time:.6f} seconds")
    
    # Compare results if all algorithms completed successfully
    if 'johnson_distances' in results and 'bellman_ford_distances' in results:
        # Check if results match
        johnson_dist = results['johnson_distances']
        bf_dist = results['bellman_ford_distances']
        
        # Compare distances
        all_match = True
        for node in G.nodes():
            if node in johnson_dist and node in bf_dist:
                if abs(johnson_dist[node] - bf_dist[node]) > 1e-6:
                    all_match = False
                    print(f"Discrepancy for node {node}: Johnson = {johnson_dist[node]}, Bellman-Ford = {bf_dist[node]}")
        
        if all_match:
            print("\nResults from Johnson's algorithm and Bellman-Ford match.")
        else:
            print("\nWarning: Results from Johnson's algorithm and Bellman-Ford differ!")
    
    return results

def generate_random_graph(n_nodes, edge_density=0.3, min_weight=-10, max_weight=20, allow_negative_cycles=False):
    """
    Generate a random directed graph
    
    Args:
        n_nodes: Number of nodes
        edge_density: Probability of edge between any two nodes
        min_weight: Minimum edge weight
        max_weight: Maximum edge weight
        allow_negative_cycles: If False, ensure no negative cycles
        
    Returns:
        NetworkX DiGraph
    """
    G = nx.DiGraph()
    
    # Add nodes
    for i in range(n_nodes):
        G.add_node(i)
    
    # Add random edges
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j and random.random() < edge_density:
                weight = random.randint(min_weight, max_weight)
                G.add_edge(i, j, weight=weight)
    
    # If negative cycles are not allowed, check and modify graph
    if not allow_negative_cycles:
        has_cycle, cycle = detect_negative_cycle(G)
        while has_cycle:
            # Break one edge in the cycle by removing it or making it positive
            u, v = cycle[0], cycle[1]
            if random.random() < 0.5:
                G.remove_edge(u, v)
            else:
                G[u][v]['weight'] = abs(G[u][v]['weight'])
            
            # Check again
            has_cycle, cycle = detect_negative_cycle(G)
    
    return G

def perform_scaling_analysis(max_nodes=50, step=5, trials=3):
    """
    Analyze how the algorithms scale with graph size
    
    Args:
        max_nodes: Maximum number of nodes to test
        step: Step size for increasing node count
        trials: Number of trials for each graph size
    """
    nodes = list(range(10, max_nodes + 1, step))
    johnson_times = []
    bellman_ford_times = []
    dijkstra_times = []
    
    print("\nPerforming scaling analysis...")
    
    for n in nodes:
        print(f"\nTesting with {n} nodes...")
        
        j_time_total = 0
        bf_time_total = 0
        d_time_total = 0
        
        for t in range(trials):
            # Generate random graph without negative cycles
            G = generate_random_graph(n, edge_density=0.3, min_weight=-5, max_weight=10, allow_negative_cycles=False)
            source = 0
            
            # Test Johnson's algorithm
            start_time = time.time()
            johnson_shortest_paths(G, source)
            j_time = time.time() - start_time
            j_time_total += j_time
            
            # Test Bellman-Ford algorithm
            start_time = time.time()
            find_shortest_paths(G, source)
            bf_time = time.time() - start_time
            bf_time_total += bf_time
            
            # Test Dijkstra's algorithm on graph with only positive edges
            G_pos = G.copy()
            for u, v, d in G.edges(data=True):
                if d['weight'] < 0:
                    G_pos[u][v]['weight'] = abs(d['weight'])
            
            start_time = time.time()
            direct_dijkstra(G_pos, source)
            d_time = time.time() - start_time
            d_time_total += d_time
            
            print(f"  Trial {t+1}/{trials} - Johnson: {j_time:.6f}s, Bellman-Ford: {bf_time:.6f}s, Dijkstra: {d_time:.6f}s")
        
        # Calculate average times
        johnson_times.append(j_time_total / trials)
        bellman_ford_times.append(bf_time_total / trials)
        dijkstra_times.append(d_time_total / trials)
        
        print(f"Average times for {n} nodes - Johnson: {johnson_times[-1]:.6f}s, Bellman-Ford: {bellman_ford_times[-1]:.6f}s, Dijkstra: {dijkstra_times[-1]:.6f}s")
    
    # Visualize results
    visualize_performance_comparison(nodes, johnson_times, bellman_ford_times, dijkstra_times)
    
    # Return data for further analysis
    return {
        'nodes': nodes,
        'johnson_times': johnson_times,
        'bellman_ford_times': bellman_ford_times,
        'dijkstra_times': dijkstra_times
    }

def visualize_time_comparison(results, title="Algorithm Timing Comparison"):
    """
    Visualize time comparison between algorithms for a specific graph
    
    Args:
        results: Dictionary with timing results
        title: Title for the plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Prepare data
    algorithms = []
    times = []
    
    if 'johnson_time' in results:
        algorithms.append("Johnson's")
        times.append(results['johnson_time'])
    
    if 'bellman_ford_time' in results:
        algorithms.append("Bellman-Ford")
        times.append(results['bellman_ford_time'])
    
    if 'dijkstra_time' in results:
        algorithms.append("Dijkstra's")
        times.append(results['dijkstra_time'])
    
    # Create bar chart
    bars = ax.bar(algorithms, times, color=['blue', 'red', 'green'][:len(algorithms)])
    
    # Add time labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{height:.6f}s',
                ha='center', va='bottom', rotation=0)
    
    ax.set_ylabel('Execution Time (seconds)')
    ax.set_title(title)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('algorithm_timing.png', dpi=300)
    plt.show()

# ---------------------------
# Main Function with Extended Functionality
# ---------------------------

def main():
    # Create test graph with negative edges
    G = nx.DiGraph()
    G.add_weighted_edges_from([
        ('A', 'B', -84),
        ('A', '4', -14),
        ('A', 'C', 2),
        ('B', 'C', -3),
        ('B', 'D', 2),
        ('C', 'D', -4),
        ('C', 'E', 5),
        ('D', 'E', -5),
        ('E', 'B', 3), 
        ('F', 'A', -7),
        ('F', 'C', 10),
        ('G', 'F', -2)  # Creates a negative cycle
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
        
        # Step 1: Decompose into strongly connected components
        print("\nDecomposing graph into strongly connected components (SCCs):")
        scc_nodes = list(nx.strongly_connected_components(G))
        scc_subgraphs = decompose_into_sccs(G)
        
        # Visualize components
        visualize_components(G, scc_subgraphs)
        
        # Step 2: Create condensation graph
        print("\nCreating condensation graph:")
        C = create_condensation_graph(G, scc_nodes)
        
        # Visualize condensation graph
        visualize_condensation(G, C, scc_nodes)
        
        # Step 3: Apply Johnson's algorithm to the graph
        print("\nAttempting to apply Johnson's algorithm to original graph with negative cycle:")
        reweighted_G, potentials = johnson_reweight(G)
        
        if reweighted_G is not None:
            # Visualize reweighted graph
            visualize_graphs(G, reweighted_G=reweighted_G, potentials=potentials, 
                            title="Johnson's Reweighting Algorithm")
        
        # Step 4: Modify graph to break negative cycle for comparison
        print("\nModifying graph to remove negative cycle:")
        G_modified = G.copy()
        G_modified.remove_edge('E', 'B')  # Remove the edge that completes the negative cycle
        
        # Visualize original and modified graphs
        visualize_graphs(G, modified_G=G_modified, title="Breaking the Negative Cycle")
        
        # Verify no negative cycles in the modified graph
        has_cycle_modified, cycle_modified = detect_negative_cycle(G_modified)
        if not has_cycle_modified:
            print("Successfully removed negative cycle.")
            
            # Step 5: Performance comparison on modified graph
            print("\n===== PERFORMANCE COMPARISON =====")
            print("Comparing algorithm performance on the modified graph (no negative cycles):")
            source = 'A'
            
            # Compare algorithm performance
            results = compare_algorithm_performance(G_modified, source)
            
            if results:
                # Visualize timing comparison
                visualize_time_comparison(results, 
                                         title=f"Execution Time Comparison for Finding Shortest Paths from {source}")
                
                # Additional analysis: show specific paths
                print("\nShortest paths from", source)
                print("-" * 40)
                
                # Johnson's results
                if 'johnson_distances' in results:
                    johnson_dist = results['johnson_distances']
                    print("\nJohnson's algorithm distances:")
                    for node in sorted(johnson_dist.keys()):
                        print(f"{source} -> {node}: {johnson_dist[node]}")
                
                # Bellman-Ford results
                if 'bellman_ford_distances' in results:
                    bf_dist = results['bellman_ford_distances']
                    print("\nBellman-Ford algorithm distances:")
                    for node in sorted(bf_dist.keys()):
                        print(f"{source} -> {node}: {bf_dist[node]}")
            
            # Step 6: Apply Johnson's algorithm to modified graph and visualize
            print("\nApplying Johnson's algorithm to the modified graph:")
            johnson_result = johnson_shortest_paths(G_modified, source)
            
            if johnson_result:
                # Visualize reweighted graph
                visualize_graphs(G_modified, reweighted_G=johnson_result['reweighted_graph'], 
                                potentials=johnson_result['potentials'], 
                                title="Johnson's Algorithm on Modified Graph")
                
                # Step 7: Compare with Bellman-Ford shortest paths
                print("\nFinding shortest paths with Bellman-Ford algorithm:")
                bf_distances, bf_predecessors = find_shortest_paths(G_modified, source)
                
                if bf_distances:
                    # Visualize Bellman-Ford shortest paths
                    visualize_graphs(G_modified, source, bf_distances, bf_predecessors, 
                                     title=f"Bellman-Ford Shortest Paths from {source}")
            
            # Step 8: Perform scaling analysis
            print("\nDo you want to perform scaling analysis? This may take some time. (y/n)")
            response = input()
            
            if response.lower() == 'y':
                scaling_results = perform_scaling_analysis(max_nodes=50, step=5, trials=3)
                
                print("\nScaling analysis complete. Results:")
                nodes = scaling_results['nodes']
                for i, n in enumerate(nodes):
                    print(f"Nodes: {n}, Johnson: {scaling_results['johnson_times'][i]:.6f}s, " + 
                          f"Bellman-Ford: {scaling_results['bellman_ford_times'][i]:.6f}s, " +
                          f"Dijkstra: {scaling_results['dijkstra_times'][i]:.6f}s")
        else:
            print(f"Failed to remove negative cycle: {' -> '.join(str(n) for n in cycle_modified)}")
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

    print("\nGraph visualizations saved as PNG files.")

if __name__ == "__main__":
    main()