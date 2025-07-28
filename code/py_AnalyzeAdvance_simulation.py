#!/usr/bin/env python3
"""
py_AnalyzeAdvance_simulation.py

Computes and exports the ProbabilityPositiveAdvance for geographic routing
in an ad-hoc airborne network.

Performs:
  - Monte Carlo over a range of node counts and repetition runs.
  - For each (NumNodes, Rep, Radius), extracts P(advance > 0).
Outputs:
  - A single CSV: ProbabilityPositiveAdvance_{max_num_nodes}.csv
    containing columns [NumNodes, Rep, Radius, ProbabilityPositiveAdvance].
"""

import os
import sys
import csv
from itertools import product
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
from scipy.stats import binned_statistic_2d
from scipy.stats import norm

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
import matplotlib.ticker as ticker
import os
import sys
from sklearn.metrics.pairwise import euclidean_distances
from multiprocessing import Pool, cpu_count
from itertools import product
import csv
from collections import deque, defaultdict
import time
import matplotlib.cm as cm
import seaborn as sns
from scipy.stats import expon
from scipy.stats import pearsonr
import math

from confidence_intervals import confidence_interval_init

from matplotlib.lines import Line2D

plt.rcParams.update({
    'font.family': 'lmodern',
    # "font.serif": 'Times',
    'font.size': 30,
    'text.usetex': True,
    'pgf.rcfonts': False,
    # 'figure.dpi': 300,
    'savefig.dpi': 300,
    'text.latex.preamble': r'\usepackage{lmodern}'
})

def create_network(area_length, area_side, num_nodes, a2a_comm_range, a2g_comm_range, GS_id, GS_pos):
    # Create positions in a dictionary format
    # positions = {i: (random.uniform(0, area_length), random.uniform(0, area_side)) for i in range(num_nodes)}
    positions = {i: (np.random.uniform(0, area_length), np.random.uniform(0, area_side)) for i in range(num_nodes)}

    # Initialize the graph
    G = nx.Graph()
    G.add_nodes_from(positions.keys())
    for i, pos in positions.items():
        for j, pos2 in positions.items():
            if i < j and np.hypot(pos[0] - pos2[0], pos[1] - pos2[1]) <= a2a_comm_range:
                G.add_edge(i, j)

    # Add the GS_id node with its specified position
    G.add_node(GS_id, pos=GS_pos)
    positions[GS_id] = GS_pos  # Include GS position in the positions dictionary

    # Connect GS_id to other nodes based on a2g_comm_range
    for i, pos in positions.items():
        if i != GS_id and np.hypot(GS_pos[0] - pos[0], GS_pos[1] - pos[1]) <= a2g_comm_range:
            G.add_edge(GS_id, i)

    return G, positions

# def process_all_nodes_with_distance(G, positions, radius=7, GS_id=501):
#     """
#     Applies analysis to every node except the destination node in the graph G,
#     returning neighbor details and Euclidean distance to the specified destination node (GS_id).

#     Parameters:
#     ----------
#     G : graph
#         A NetworkX Graph or DiGraph.
#     radius : int, optional
#         Include all neighbors of distance <= radius from each node (default is 7).
#     GS_id : int
#         The node id of the destination node for distance calculations.

#     Returns:
#     -------
#     dict: A dictionary with each node as keys except GS_id. Each value contains:
#         - neighbors_info: Dictionary containing neighbor information classified by hop distance.
#         - euclidean_distance_to_GS: The Euclidean distance from the node to GS_id.
#     """
#     all_results = {}
#     for n in G.nodes():
#         if n == GS_id:
#             continue  # Skip processing for the destination node

#         # Calculate shortest paths and determine neighbors up to the specified radius
#         sp_paths = nx.single_source_shortest_path(G, n, cutoff=radius)

#         # Collect neighbor information by hop count (from 1-hop to 7-hop)
#         neighbors_info = {f'{i}-hop': [] for i in range(1, radius + 1)}
#         for node, path in sp_paths.items():
#             if node != n and node != GS_id:  # Exclude the center and GS_id from neighbor info
#                 num_hops = len(path) - 1
#                 if num_hops <= radius:
#                     via_node = path[1] if len(path) > 1 else None
#                     hop_key = f'{num_hops}-hop'
#                     neighbors_info[hop_key].append((node, via_node))

#         # Compute Euclidean distance to the destination node (GS_id)
#         gs_pos = positions[GS_id]
#         n_pos = positions[n]
#         distance_to_GS = np.linalg.norm(np.array(n_pos) - np.array(gs_pos))

#         # Store the results including the distance to GS
#         all_results[n] = {
#             'neighbors_info': neighbors_info,
#             'distance_to_GS': distance_to_GS
#         }

#     return all_results
def process_all_nodes_with_distance(G, positions, radius=7, GS_id=501):
    """
    Applies analysis to every node except the destination node in the graph G,
    returning neighbor details and Euclidean distance to the specified destination node (GS_id),
    using all shortest paths to find the farthest via node.

    Parameters:
    ----------
    G : graph
        A NetworkX Graph or DiGraph.
    radius : int, optional
        Include all neighbors of distance <= radius from each node (default is 7).
    GS_id : int
        The node id of the destination node for distance calculations.

    Returns:
    -------
    dict: A dictionary with each node as keys except GS_id. Each value contains:
        - neighbors_info: Dictionary containing neighbor information classified by hop distance.
        - euclidean_distance_to_GS: The Euclidean distance from the node to GS_id.
    """
    all_results = {}
    precomputed_distances = {
        (n1, n2): np.linalg.norm(np.array(positions[n1]) - np.array(positions[n2]))
        for n1 in G.nodes for n2 in G.nodes
    }

    for n in G.nodes():
        if n == GS_id:
            continue  # Skip processing for the destination node

        # Calculate shortest paths and determine neighbors up to the specified radius
        sp_paths = nx.single_source_shortest_path(G, n, cutoff=radius)

        # Collect neighbor information by hop count (from 1-hop to 7-hop)
        neighbors_info = {f'{i}-hop': [] for i in range(1, radius + 1)}
        for node, path in sp_paths.items():
            if node != n and node != GS_id:  # Exclude the center and GS_id from neighbor info
                num_hops = len(path) - 1
                if num_hops <= radius:
                    # Check if there are multiple shortest paths
                    if len(path) > 2 and nx.has_path(G, n, node):
                        all_paths = list(nx.all_shortest_paths(G, source=n, target=node))
                    else:
                        all_paths = [path]  # Use the single shortest path
                    
                    # Find the farthest via node
                    farthest_via_node = max(
                        (path[1] for path in all_paths if len(path) > 1),
                        key=lambda via: precomputed_distances[(n, via)],
                        default=None
                    )
                    
                    hop_key = f'{num_hops}-hop'
                    neighbors_info[hop_key].append((node, farthest_via_node))

        # Compute Euclidean distance to the destination node (GS_id)
        gs_pos = positions[GS_id]
        n_pos = positions[n]
        distance_to_GS = np.linalg.norm(np.array(n_pos) - np.array(gs_pos))

        # Store the results including the distance to GS
        all_results[n] = {
            'neighbors_info': neighbors_info,
            'distance_to_GS': distance_to_GS
        }

    return all_results

def calculate_k_hop_neighbors(all_results, k):
    """
    Calculates the average number of k-hop neighbors across all nodes.
    
    Parameters:
    ----------
    all_results : dict
        The node information containing neighbors for each node.
    k : int
        The number of hops to consider.

    Returns:
    -------
    float: The average number of k-hop neighbors for all nodes.
    """
    total_k_hop_neighbors = 0
    count = 0
    for node in all_results:
        total_k_hop_neighbors += len(all_results[node]['neighbors_info'][f'{k}-hop'])
        count += 1
    return total_k_hop_neighbors / count if count > 0 else 0

def best_advance_within_k_hops(current, all_results, a2a_comm_range, k):
    """
    Finds the best node to advance within `k` hops.
    
    Parameters:
    ----------
    current : int
        The current node.
    all_results : dict
        A dictionary containing details for each node.
    a2a_comm_range : float
        The communication range for A2A communication.
    k : int
        The number of hops to consider.

    Returns:
    -------
    int: The node to advance towards, or None if no advancement found.
    """
    current_to_dest_distance = all_results[current]['distance_to_GS']
    best_node = None
    best_advance = k * a2a_comm_range  # Initialize with maximum advancement possible
    
    # Combine all hops up to k-hops into a single loop
    neighbors = (neighbor_info for hop_type in [f'{i}-hop' for i in range(1, k+1)]
                                for neighbor_info in all_results[current]['neighbors_info'][hop_type])
    
    for neighbor, via_node in neighbors:
        neighbor_to_dest_distance = all_results[neighbor]['distance_to_GS']
        # Calculate the geographic advancement
        advance = current_to_dest_distance - neighbor_to_dest_distance + k * a2a_comm_range

        if advance > best_advance:
            best_advance = advance
            best_node = via_node  # Use the via node as the best node to advance toward

    return best_node

# def perform_geographic_routing(G, all_results, a2a_comm_range, a2g_comm_range, destination, find_next_hop, k):
#     """
#     Perform greedy geographic routing from each node in the graph to a destination
#     node and calculate the hop stretch factor compared to Dijkstra.
    
#     Parameters:
#     ----------
#     G : networkx.Graph
#         The graph over which nodes are routed.
#     all_results : dict
#         A dictionary containing details for each node.
#     a2a_comm_range : float
#         The communication range for A2A communication.
#     a2g_comm_range : float
#         The communication range for A2G communication.
#     destination : int
#         The destination node.
#     find_next_hop : callable
#         Function used to determine the best next hop.
#     k : int
#         The number of hops to consider.

#     Returns:
#     -------
#     tuple: 
#         - int: The number of successful routes.
#         - list: The hop stretch factors for each node with a valid path.
#     """
#     successful_routes = 0
#     hop_stretch_factors = []  # To store the stretch factor for valid paths
#     path_cache = {}  # Cache for successful paths
#     failed_path_cache = {}  # Cache for paths known to fail

#     for node in G.nodes:
#         if node != destination:

#             # Check if a path exists using Dijkstra's algorithm
#             try:
#                 dijkstra_path = nx.dijkstra_path(G, node, destination)
#                 dijkstra_path_length = len(dijkstra_path) - 1  # Hop count in Dijkstra path
#             except nx.NetworkXNoPath:
#                 failed_path_cache[(node, destination)] = True  # Add original node to failed path cache
#                 continue  # Move to the next node since no path exists

#             current = node
#             path = [current]  # Initialize path with the current node
#             greedy_path_length = 0

#             # Compute the path if not in cache
#             while current != destination:
#                 path_key = (current, destination)

#                 # Check failed path cache first
#                 if path_key in failed_path_cache:
#                     failed_path_cache[(node, destination)] = True  # Add original node to failed path cache
#                     break  # If intermediate path is known to fail, stop processing

#                 # Attempt to use the successful path cache
#                 if path_key in path_cache:
#                     path.extend(path_cache[path_key])  # Extend path with cached successful route
#                     greedy_path_length = len(path) - 1  # Hop count in the greedy path
#                     successful_routes += 1
#                     hop_stretch_factors.append(greedy_path_length / dijkstra_path_length)  # Compute hop stretch
#                     path_cache[(node, destination)] = path  # Cache the full path from the original node
#                     break  # If path is known to succeed, stop processing

#                 # Check if direct connection to destination is possible within communication range
#                 distance_to_GS = all_results[current]['distance_to_GS']
#                 if distance_to_GS <= a2g_comm_range:
#                     path.append(destination)  # Directly add the destination since it's within range
#                     greedy_path_length = len(path) - 1  # Hop count in the greedy path
#                     successful_routes += 1
#                     hop_stretch_factors.append(greedy_path_length / dijkstra_path_length)  # Include last hop stretch
#                     path_cache[(node, destination)] = path  # Cache the full path from the original node

#                     # Cache the full path for each node in the path
#                     for i in range(len(path) - 1):
#                         subpath_key = (path[i], destination)
#                         if subpath_key not in path_cache:
#                             path_cache[subpath_key] = path[i + 1:]  # Cache the subpath from node i to destination
#                     break

#                 next_hop = find_next_hop(current, all_results, a2a_comm_range, k)
#                 if next_hop is None or next_hop in path:
#                     failed_path_cache[path_key] = True  # Mark this as a failed route
#                     for failed_node in path:
#                         failed_path_cache[(failed_node, destination)] = True  # Mark all nodes in path as failed
#                     break
#                 path.append(next_hop)
#                 current = next_hop
#                 greedy_path_length += 1

#             # If we reached the destination through greedy routing (not directly via A2G), record the stretch factor
#             if current == destination:
#                 greedy_path_length = len(path) - 1  # Hop count in the greedy path
#                 successful_routes += 1
#                 hop_stretch_factors.append(greedy_path_length / dijkstra_path_length)  # Compute hop stretch

#                 # Cache the full path for each node in the path
#                 for i in range(len(path) - 1):
#                     subpath_key = (path[i], destination)
#                     if subpath_key not in path_cache:
#                         path_cache[subpath_key] = path[i + 1:]  # Cache the subpath from node i to destination

#     return successful_routes, hop_stretch_factors

# def perform_geographic_routing(G, all_results, a2a_comm_range, a2g_comm_range, destination, find_next_hop, k):
#     """
#     Perform greedy geographic routing from each node in the graph to a destination
#     node and calculate the hop stretch factor compared to Dijkstra.
    
#     Parameters:
#     ----------
#     G : networkx.Graph
#         The graph over which nodes are routed.
#     all_results : dict
#         A dictionary containing details for each node.
#     a2a_comm_range : float
#         The communication range for A2A communication.
#     a2g_comm_range : float
#         The communication range for A2G communication.
#     destination : int
#         The destination node.
#     find_next_hop : callable
#         Function used to determine the best next hop.
#     k : int
#         The number of hops to consider.

#     Returns:
#     -------
#     tuple: 
#         - int: The number of successful routes.
#         - list: The hop stretch factors for each node with a valid path.
#     """
#     successful_routes = 0
#     hop_stretch_factors = []  # To store the stretch factor for valid paths
#     path_cache = {}  # Cache for successful paths
#     failed_path_cache = {}  # Cache for paths known to fail
#     num_nodes = G.number_of_nodes() - 1


#     # Initialize DataFrame to store results
#     # routing_df = pd.DataFrame(columns=["Node", "DijkstraNextHops", "GeographicNextHop", "Exists"])
#     routing_df = pd.DataFrame(columns=["Node", "DijkstraNextHops", "GeographicNextHop", "Exists", "TopologicalAdvance"])

#     for node in G.nodes:
#         if node != destination:

#             # # Check if a path exists using Dijkstra's algorithm
#             # try:
#             #     dijkstra_path = nx.dijkstra_path(G, node, destination)
#             #     dijkstra_path_length = len(dijkstra_path) - 1  # Hop count in Dijkstra path
#             # except nx.NetworkXNoPath:
#             #     failed_path_cache[(node, destination)] = True  # Add original node to failed path cache
#             #     continue  # Move to the next node since no path exists
#             # Check if a path exists using Dijkstra's algorithm
#             # Check if a path exists using Dijkstra's algorithm
#             try:
#                 all_dijkstra_paths = list(nx.all_shortest_paths(G, node, destination))
#                 dijkstra_next_hops = list(set(path[1] for path in all_dijkstra_paths))
#                 dijkstra_path_length = len(all_dijkstra_paths[0]) - 1  # Hop count in Dijkstra path

#                 routing_df = pd.concat([
#                     routing_df,
#                     pd.DataFrame({
#                         "Node": [node],
#                         "DijkstraNextHops": [dijkstra_next_hops],
#                         "GeographicNextHop": [None],
#                         "Exists": [0],
#                         "TopologicalAdvance": [0]
#                     })
#                 ], ignore_index=True)

#             except nx.NetworkXNoPath:
#                 dijkstra_next_hops = None
#                 dijkstra_path_length = None
#                 geographic_next_hop = None
#                 exists = 0
#                 routing_df = pd.concat([
#                     routing_df,
#                     pd.DataFrame({
#                         "Node": [node],
#                         "DijkstraNextHops": [dijkstra_next_hops],
#                         "GeographicNextHop": [None],
#                         "Exists": [0],
#                         "TopologicalAdvance": [0]
#                     })
#                 ], ignore_index=True)
#                 failed_path_cache[(node, destination)] = True  # Add original node to failed path cache
#                 continue  # Move to the next node since no path exists
    
#             current = node
#             path = [current]  # Initialize path with the current node
#             greedy_path_length = 0

#             # Compute the path if not in cache
#             while current != destination:
#                 path_key = (current, destination)

#                 # Check failed path cache first
#                 if path_key in failed_path_cache:
#                     failed_path_cache[(node, destination)] = True  # Add original node to failed path cache
#                     break  # If intermediate path is known to fail, stop processing

#                 # Attempt to use the successful path cache
#                 if path_key in path_cache:
#                     path.extend(path_cache[path_key])  # Extend path with cached successful route
#                     greedy_path_length = len(path) - 1  # Hop count in the greedy path
#                     successful_routes += 1
#                     hop_stretch_factors.append(greedy_path_length / dijkstra_path_length)  # Compute hop stretch
#                     path_cache[(node, destination)] = path  # Cache the full path from the original node
#                     break  # If path is known to succeed, stop processing

#                 # Check if direct connection to destination is possible within communication range
#                 distance_to_GS = all_results[current]['distance_to_GS']
#                 if distance_to_GS <= a2g_comm_range:
#                     path.append(destination)  # Directly add the destination since it's within range
#                     greedy_path_length = len(path) - 1  # Hop count in the greedy path
#                     successful_routes += 1
#                     hop_stretch_factors.append(greedy_path_length / dijkstra_path_length)  # Include last hop stretch
#                     path_cache[(node, destination)] = path  # Cache the full path from the original node

#                     # Cache the full path for each node in the path
#                     for i in range(len(path) - 1):
#                         subpath_key = (path[i], destination)
#                         if subpath_key not in path_cache:
#                             path_cache[subpath_key] = path[i + 1:]  # Cache the subpath from node i to destination
#                     break

#                 next_hop = find_next_hop(current, all_results, a2a_comm_range, k)
#                 if next_hop is None or next_hop in path:
#                     failed_path_cache[path_key] = True  # Mark this as a failed route
#                     for failed_node in path:
#                         failed_path_cache[(failed_node, destination)] = True  # Mark all nodes in path as failed
                    
#                     topological_advance = 0
#                     if next_hop is not None:
#                         # Calculate topological advance
#                         current_distance = all_results[current]['distance_to_GS']
#                         next_hop_distance = all_results[next_hop]['distance_to_GS']
#                         current_advance = calculate_topological_advance(current_distance, a2a_comm_range, a2g_comm_range)
#                         next_hop_advance = calculate_topological_advance(next_hop_distance, a2a_comm_range, a2g_comm_range)
#                         topological_advance = next_hop_advance == current_advance - 1

#                     # Update DataFrame for failed or invalid cases
#                     routing_df = pd.concat([
#                         routing_df,
#                         pd.DataFrame({
#                             "Node": [node],
#                             "DijkstraNextHops": [dijkstra_next_hops],
#                             "GeographicNextHop": [next_hop if next_hop is not None else None],
#                             "Exists": [1 if next_hop is not None and next_hop in dijkstra_next_hops else 0],
#                             "TopologicalAdvance": [1 if topological_advance else 0],
#                         })
#                     ], ignore_index=True)
                    
#                     break

#                 path.append(next_hop)
#                 current = next_hop
#                 greedy_path_length += 1

#             # If we reached the destination through greedy routing (not directly via A2G), record the stretch factor
#             if current == destination:
#                 greedy_path_length = len(path) - 1  # Hop count in the greedy path
#                 successful_routes += 1
#                 hop_stretch_factors.append(greedy_path_length / dijkstra_path_length)  # Compute hop stretch

#                 # Cache the full path for each node in the path
#                 for i in range(len(path) - 1):
#                     subpath_key = (path[i], destination)
#                     if subpath_key not in path_cache:
#                         path_cache[subpath_key] = path[i + 1:]  # Cache the subpath from node i to destination
    
#     # Update the DataFrame by iterating over the path cache
#     for (node, destination), path in path_cache.items():
#         topological_advance = 0
#         next_hop_distance = 0
#         if len(path) > 0:
#             geographic_next_hop = path[1]
#             current_distance = all_results[node]['distance_to_GS']
#             if geographic_next_hop == destination:
#                 next_hop_distance = 0
#             else:
#                 next_hop_distance = all_results[geographic_next_hop]['distance_to_GS']
#             current_advance = calculate_topological_advance(current_distance, a2a_comm_range, a2g_comm_range)
#             next_hop_advance = calculate_topological_advance(next_hop_distance, a2a_comm_range, a2g_comm_range)
#             topological_advance = next_hop_advance == current_advance - 1
#         else:
#             geographic_next_hop = None



#         # Retrieve the Dijkstra next hops from the DataFrame
#         dijkstra_next_hops_row = routing_df[routing_df["Node"] == node]
#         dijkstra_next_hops = dijkstra_next_hops_row["DijkstraNextHops"].iloc[0] if not dijkstra_next_hops_row.empty else None

#         # Compute whether the geographic next hop exists in the Dijkstra next hops
#         exists = 1 if geographic_next_hop in (dijkstra_next_hops or []) else 0

#         # Add to the DataFrame
#         routing_df = pd.concat([
#             routing_df,
#             pd.DataFrame({
#                 "Node": [node],
#                 "DijkstraNextHops": [dijkstra_next_hops],
#                 "GeographicNextHop": [geographic_next_hop],
#                 "Exists": [exists],
#                 "TopologicalAdvance": [1 if topological_advance else 0]
#             })
#         ], ignore_index=True)

#     # Drop duplicates based on Node, keeping the last occurrence
#     routing_df.drop_duplicates(subset=["Node"], keep="last", inplace=True)

#     # Compute probability_zero_advance, including rows with None in DijkstraNextHops
#     probability_zero_advance = routing_df["TopologicalAdvance"].mean()
    
#     # Filter out rows where DijkstraNextHops is None
#     routing_df = routing_df[routing_df["DijkstraNextHops"].notna()]

#     # Reset the index after filtering
#     routing_df.reset_index(drop=True, inplace=True)

#     # Compute probability_positive_advance after filtering
#     probability_positive_advance = routing_df["Exists"].mean()

#     return successful_routes, hop_stretch_factors, probability_positive_advance, probability_zero_advance
def perform_geographic_routing(G, all_results, a2a_comm_range, a2g_comm_range, destination, find_next_hop, k):
    """
    Perform greedy geographic routing from each node in the graph to a destination
    node and calculate the hop stretch factor compared to Dijkstra.
    
    Parameters:
    ----------
    G : networkx.Graph
        The graph over which nodes are routed.
    all_results : dict
        A dictionary containing details for each node.
    a2a_comm_range : float
        The communication range for A2A communication.
    a2g_comm_range : float
        The communication range for A2G communication.
    destination : int
        The destination node.
    find_next_hop : callable
        Function used to determine the best next hop.
    k : int
        The number of hops to consider.

    Returns:
    -------
    tuple: 
        - int: The number of successful routes.
        - list: The hop stretch factors for each node with a valid path.
    """
    successful_routes = 0
    hop_stretch_factors = []  # To store the stretch factor for valid paths
    path_cache = {}  # Cache for successful paths
    failed_path_cache = {}  # Cache for paths known to fail
    num_nodes = G.number_of_nodes() - 1
    next_hop_dict = {}
   
    # Compute hop counts from each node to the destination using Dijkstra
    hop_counts_to_dest = nx.single_source_shortest_path_length(G, destination)
    
    # Initialize the next hop dictionary with -1 for all nodes except the destination
    next_hop_dict = {node: -1 for node in hop_counts_to_dest if node != destination}

    for node in G.nodes:
        if node != destination:

            # # Check if a path exists using Dijkstra's algorithm
            # try:
            #     dijkstra_path = nx.dijkstra_path(G, node, destination)
            #     dijkstra_path_length = len(dijkstra_path) - 1  # Hop count in Dijkstra path
            # except nx.NetworkXNoPath:
            #     failed_path_cache[(node, destination)] = True  # Add original node to failed path cache
            #     continue  # Move to the next node since no path exists
            # Check if a path exists using Dijkstra's algorithm
            # Check if a path exists using Dijkstra's algorithm
            try:
                all_dijkstra_paths = list(nx.all_shortest_paths(G, node, destination))
                dijkstra_next_hops = list(set(path[1] for path in all_dijkstra_paths))
                dijkstra_path_length = len(all_dijkstra_paths[0]) - 1  # Hop count in Dijkstra path

            except nx.NetworkXNoPath:
                dijkstra_next_hops = None
                dijkstra_path_length = None
                geographic_next_hop = None
                failed_path_cache[(node, destination)] = True  # Add original node to failed path cache
                continue  # Move to the next node since no path exists
    
            current = node
            path = [current]  # Initialize path with the current node
            greedy_path_length = 0

            # Compute the path if not in cache
            while current != destination:
                path_key = (current, destination)

                # Check failed path cache first
                if path_key in failed_path_cache:
                    failed_path_cache[(node, destination)] = True  # Add original node to failed path cache
                    break  # If intermediate path is known to fail, stop processing

                # Attempt to use the successful path cache
                if path_key in path_cache:
                    # path.extend(path_cache[path_key])  # Extend path with cached successful route
                    # print(f"path: {path}, path_key: {path_key}")
                    cached_subpath = path_cache[path_key]
                    # If the last node in 'path' is the same as the first node in 'cached_subpath',
                    # we skip the first element to avoid duplication:
                    if path and cached_subpath and path[-1] == cached_subpath[0]:
                        path.extend(cached_subpath[1:])
                    else:
                        path.extend(cached_subpath)
                    if len(path) != len(set(path)):
                        print("(A) When you find a cached subpath and extend")
                        print("Warning: path has duplicates right before storing:", path)
                    # print(ss)
                    greedy_path_length = len(path) - 1  # Hop count in the greedy path
                    successful_routes += 1
                    # hop_stretch_factors.append(greedy_path_length / dijkstra_path_length)  # Compute hop stretch
                    path_cache[(node, destination)] = path  # Cache the full path from the original node
                    break  # If path is known to succeed, stop processing

                # Check if direct connection to destination is possible within communication range
                distance_to_GS = all_results[current]['distance_to_GS']
                if distance_to_GS <= a2g_comm_range:
                    path.append(destination)  # Directly add the destination since it's within range
                    greedy_path_length = len(path) - 1  # Hop count in the greedy path
                    successful_routes += 1
                    # hop_stretch_factors.append(greedy_path_length / dijkstra_path_length)  # Include last hop stretch
                    path_cache[(node, destination)] = path  # Cache the full path from the original node
                    if len(path) != len(set(path)):
                        print("(B) When you have direct A2G communication:")
                        print("Warning: path has duplicates right before storing:", path)
                    # Cache the full path for each node in the path
                    for i in range(len(path) - 1):
                        subpath_key = (path[i], destination)
                        if subpath_key not in path_cache:
                            path_cache[subpath_key] = path[i + 1:]  # Cache the subpath from node i to destination

                    if destination and current in hop_counts_to_dest:  # Ensure the next hop exists and has a valid path
                        # Update the dictionary with the next hop
                        next_hop_dict[current] = destination
                    break

                next_hop = find_next_hop(current, all_results, a2a_comm_range, k)
                
                if next_hop and current in hop_counts_to_dest:  # Ensure the next hop exists and has a valid path
                    # Update the dictionary with the next hop
                    next_hop_dict[current] = next_hop

                if next_hop is None or next_hop in path:
                    failed_path_cache[path_key] = True  # Mark this as a failed route
                    for failed_node in path:
                        failed_path_cache[(failed_node, destination)] = True  # Mark all nodes in path as failed                    
                    break

                path.append(next_hop)
                current = next_hop
                greedy_path_length += 1

            # If we reached the destination through greedy routing (not directly via A2G), record the stretch factor
            if current == destination:
                greedy_path_length = len(path) - 1  # Hop count in the greedy path
                successful_routes += 1
                # hop_stretch_factors.append(greedy_path_length / dijkstra_path_length)  # Compute hop stretch

                # Cache the full path for each node in the path
                for i in range(len(path) - 1):
                    subpath_key = (path[i], destination)
                    if subpath_key not in path_cache:
                        path_cache[subpath_key] = path[i + 1:]  # Cache the subpath from node i to destination
                        if len(path) != len(set(path)):
                            print("(C) When you’ve reached destination via greedy (not direct A2G):")
                            print("Warning: path has duplicates right before storing:", path)
    # Calculate hop stretch factors using path cache
    for (start_node, _), path in path_cache.items():
        greedy_path_length = len(path) - 1
        dijkstra_path_length = hop_counts_to_dest[start_node]
        hop_stretch_factor = greedy_path_length / dijkstra_path_length
        hop_stretch_factors.append(hop_stretch_factor)

    # Step 1: Print nodes in `next_hop_dict` with valid next hops
    valid_next_hops_nodes = [node for node, next_hop in next_hop_dict.items() if next_hop != -1]

    # Step 2: Create another dictionary with hop counts of nodes and their next hops
    hop_count_advances = {}
    for node, next_hop in next_hop_dict.items():
        if next_hop != -1:  # Ensure the next hop exists
            hop_count_advances[node] = (hop_counts_to_dest[node], hop_counts_to_dest[next_hop])


    # Step 3: Calculate probabilities for positive, negative, and zero advances
    # Total nodes in next_hop_dict
    total_nodes_in_next_hop_dict = len(next_hop_dict)
    positive_advances = sum(1 for counts in hop_count_advances.values() if counts[1] < counts[0])  # Positive advance
    negative_advances = sum(1 for counts in hop_count_advances.values() if counts[1] > counts[0])  # Negative advance
    zero_advances = sum(1 for counts in hop_count_advances.values() if counts[1] == counts[0])  # Zero advance
    failures = len(next_hop_dict) - len(valid_next_hops_nodes)  # Adjusted zero advance

    # Calculate probabilities dividing by total number in next_hop_dict
    prob_positive_advance = positive_advances / total_nodes_in_next_hop_dict if total_nodes_in_next_hop_dict > 0 else 0
    prob_negative_advance = negative_advances / total_nodes_in_next_hop_dict if total_nodes_in_next_hop_dict > 0 else 0
    prob_zero_advance = zero_advances / total_nodes_in_next_hop_dict if total_nodes_in_next_hop_dict > 0 else 0
    prob_failure = failures / total_nodes_in_next_hop_dict if total_nodes_in_next_hop_dict > 0 else 0

    return successful_routes, hop_stretch_factors, prob_positive_advance, prob_zero_advance, prob_negative_advance, prob_failure

# def calculate_topological_advance(distance, a2a_comm_range, a2g_comm_range):
#     hop_range = a2a_comm_range
#     max_gs_range = a2g_comm_range
#     if distance <= max_gs_range:
#         return 1
#     return ((distance - max_gs_range - 1) // hop_range) + 2
def calculate_topological_advance(distance, a2a_comm_range, a2g_comm_range):
    if distance <= a2g_comm_range:
        return 1  # In the 1-hop region (<= a2g_comm_range)
    else:
        excess_distance = distance - a2g_comm_range
        return math.ceil(excess_distance / a2a_comm_range) + 1  # +1 to account for the 1-hop region


def run_simulation(args):
    # Unpack all necessary parameters
    num_nodes, rep, k_max = args
    
    # Set the seed for reproducibility
    np.random.seed(rep)

    # Settings and parameters
    a2a_comm_range = 100  # A2A Communication range in km
    a2g_comm_range = 370.4  # A2G Communication range in km
    area_length, area_side = 1250, 800
    GS_id = 501
    GS_pos = (1150, 400)
    # area_length, area_side = 2500, 1600
    # GS_id = 2001
    # GS_pos = (2400, 800)
    # area_length, area_side = 1768, 1131
    # GS_id = 1001
    # GS_pos = (1668, 565.5)

    # Create the network
    G, positions = create_network(area_length, area_side, num_nodes, a2a_comm_range, a2g_comm_range, GS_id, GS_pos)

    # Process nodes to get distances and neighbors (up to 7-hop)
    all_node_details = process_all_nodes_with_distance(G, positions, radius=k_max, GS_id=GS_id)

    # Scenario 1: Success Rate using Dijkstra's algorithm
    path_exists = 0
    for node in G.nodes:
        if node != GS_id and nx.has_path(G, node, GS_id):
            path_exists += 1
    success_rate_dijkstra = path_exists / num_nodes  # Exclude destination from source count

    # Scenario 2: Success Rates for Greedy-1 to Greedy-7 and hop stretch factors
    success_rates_greedy = []
    hop_stretch_factors_greedy = []
    probability_positive_advance = {}  # Dictionary to store probability of advance for each k
    probability_zero_advance = {}  # Dictionary to store probability of topological zero advance for each k
    probability_negative_advance = {}  # Dictionary to store probability of topological negative advance for each k
    probability_failure = {}  # Dictionary to store probability of failure for each k

    for k in range(1, k_max+1):  # Loop from Greedy-1 to Greedy-7
        successful_routes, hop_stretch_factors, prob_positive_advance_k, prob_zero_advance_k, prob_negative_advance_k, prob_failure_k  = perform_geographic_routing(
            G, all_node_details, a2a_comm_range, a2g_comm_range, GS_id, best_advance_within_k_hops, k
        )
        success_rate_greedy_k = successful_routes / path_exists  # Exclude destination from source count
        success_rates_greedy.append(success_rate_greedy_k)
        hop_stretch_factors_greedy.append(np.mean(hop_stretch_factors))  # Average hop stretch factor
        probability_positive_advance[k] = prob_positive_advance_k  # Add the probability of positive advance for this hop
        probability_zero_advance[k] = prob_zero_advance_k  # Add the probability of zero advance for this hop
        probability_negative_advance[k] = prob_negative_advance_k # Add the probability of negative advance for this hop
        probability_failure[k] = prob_failure_k  # Store the probability of failure for this k

    # Scenario 3: Calculate the average number of k-hop neighbors for k=1 to 7
    average_k_hop_neighbors = []
    for k in range(1, k_max+1):
        avg_neighbors = calculate_k_hop_neighbors(all_node_details, k)
        average_k_hop_neighbors.append(avg_neighbors)

    # Print the results for each repetition
    print(
        f"Completed: (Number of Nodes) {num_nodes}, "
        f"Repetition {rep}, "
        f"Success Rate (Dijkstra) {round(success_rate_dijkstra * 100, 3)}%, "
        f"Success Rates (Greedy-1 to Greedy-{k_max}) {', '.join(f'{round(rate * 100, 3)}%' for rate in success_rates_greedy)}, "
        f"Hop Stretch Factors (Greedy-1 to Greedy-{k_max}): {', '.join(f'{round(factor, 3)}' for factor in hop_stretch_factors_greedy)}, "
        f"Prob Postive Advance: {', '.join(f'{round(prob, 3)}' for prob in probability_positive_advance)}, "
        f"Prob Negative Advance: {', '.join(f'{round(prob, 3)}' for prob in probability_negative_advance)}, "
        f"Prob Zero Advance: {', '.join(f'{round(prob, 3)}' for prob in probability_positive_advance)}, "
        f"Prob Failure: {', '.join(f'{round(prob, 3)}' for prob in probability_failure)}, "

    )

    return [
        num_nodes, rep, success_rate_dijkstra, 
        success_rates_greedy, 
        hop_stretch_factors_greedy, 
        average_k_hop_neighbors, 
        probability_positive_advance,
        probability_zero_advance,
        probability_negative_advance,
        probability_failure  

    ]

if __name__ == "__main__":
    # ---------------------------------------------------------------------
    # 1) Simulation parameters
    # ---------------------------------------------------------------------
    # repititions     = 2000 # number of Monte‑Carlo runs used in the paper (HPC can easily handle this)
    repititions     = 5         # for quick local testing; paper results used 2,000 runs
    k_max           = 3
    num_node_values = np.arange(50, 501, 50)
    max_num_nodes   = 500
    equipage_fraction_values = np.divide(num_node_values, max_num_nodes)

    # 2) Directory setup for CSV output
    # ---------------------------------------------------------------------
    # 1. Find the directory this script lives in
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 2. Go one level up to the project root
    project_root = os.path.abspath(os.path.join(script_dir, os.pardir))

    # 3. Now build your results folder there
    base_csv_dir = os.path.join(project_root, "results", "csv_files")

    sim_csv_dir      = os.path.join(base_csv_dir, "simulation")
    os.makedirs(sim_csv_dir, exist_ok=True)

    simulation_csv_filename = f"simulation_greedy.csv"
    simulation_average_csv_filename = f"simulation_greedy_average.csv"
    simulation_csv_path     = os.path.join(sim_csv_dir, simulation_csv_filename)
    simulation_average_csv_path     = os.path.join(sim_csv_dir, simulation_average_csv_filename)

    # ---------------------------------------------------------------------
    # 3) Prepare Monte Carlo parameter tuples
    # ---------------------------------------------------------------------
    parameters = list(product(num_node_values, range(repititions), [k_max]))
    ncpus = int(os.environ.get('SLURM_CPUS_PER_TASK', cpu_count()))
    print(f"Using {ncpus} processes for {len(parameters)} simulations")

    # # with Pool() as pool:
    with Pool(ncpus) as pool:
        results = []
        print(f"cpus_per_task: {ncpus}")
        for result in pool.imap(run_simulation, parameters):
            results.append(result)
            # print(f"Result received: {result[:8]}")
            filtered_result = result[:5] + result[6:]  # Skipping index 4 and 5
            print(f"Result received: {filtered_result}")
        
    # Initialize a more detailed structure for storing results
    aggregated_results = {
        num_nodes: {rep: {
            'Dijkstra': None,
            **{f'Greedy-{k}': None for k in range(1, k_max+1)},
            **{f'Avg_{k}_Hop_Neighbors': None for k in range(1, k_max+1)},
            **{f'Greedy_{k}_Hop_Stretch': None for k in range(1, k_max+1)}
        } for rep in range(repititions)}
        for num_nodes in num_node_values
    }

    # Initialize a structure for storing the probability of advance
    probability_positive_advance_rows = []

    for result in results:
        num_nodes = result[0]
        rep = result[1]
        rate_dijkstra = result[2]
        rates_greedy = result[3]  # List of Greedy-1 to Greedy-7 success rates
        hop_stretch_factors = result[4]  # List of hop stretch factors for Greedy-1 to Greedy-7
        avg_k_hop_neighbors = result[5]  # List of average k-hop neighbors for 1 to 7 hops
        probability_positive_advance = result[6]  # Probability of positive advance dictionary
        probability_zero_advance = result[7]  # Probability of zero advance dictionary
        probability_negative_advance = result[8]  # Probability of negative advance dictionary
        probability_failure = result[9]  # Extract the failure probabilities

        # Add entries for the probability of advance
        for k, prob_positive_advance in probability_positive_advance.items():
            probability_positive_advance_rows.append({
                "NumNodes": num_nodes,
                "Rep": rep,
                "Radius": k,
                "GreedyRate": rates_greedy[k - 1],  # Match GreedyRate for the same radius (k)
                "HopStretchFactor": hop_stretch_factors[k - 1],  # Match HopStretchFactor for the same radius (k)
                "ProbabilityPositiveAdvance": prob_positive_advance,
                "ProbabilityZeroAdvance": probability_zero_advance[k],
                "ProbabilityNegativeAdvance": probability_negative_advance[k],
                "ProbabilityFailure": probability_failure[k]

            })

        # Store the results in the aggregated structure
        aggregated_results[num_nodes][rep] = {
            'Dijkstra': rate_dijkstra,
            **{f'Greedy-{i+1}': rates_greedy[i] for i in range(k_max)},
            **{f'Greedy_{i+1}_Hop_Stretch': hop_stretch_factors[i] for i in range(k_max)},
            **{f'Avg_{i+1}_Hop_Neighbors': avg_k_hop_neighbors[i] for i in range(k_max)}
        }

    # Create a DataFrame for probability of advance
    probability_positive_advance_df = pd.DataFrame(probability_positive_advance_rows)

    # Save the probability of advance DataFrame without aggregating
    probability_positive_advance_df.to_csv(simulation_csv_path, index=False)

    # Define the fieldnames (column headers) for the CSV
    headers = ['NumNodes', 'Avg_Dijkstra', 'MoE_Dijkstra'] + \
            [f'Avg_Greedy{k}' for k in range(1, k_max+1)] + [f'MoE_Greedy{k}' for k in range(1, k_max+1)] + \
            [f'Avg_{k}_Hop_Neighbors' for k in range(1, k_max+1)] + [f'MoE_{k}_Hop_Neighbors' for k in range(1, k_max+1)] + \
            [f'Greedy_{k}_Hop_Stretch' for k in range(1, k_max+1)] + [f'MoE_Greedy_{k}_Hop_Stretch' for k in range(1, k_max+1)]

    # Open the CSV file for writing
    with open(simulation_average_csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()  # Write the header row

        # Write data rows
        for num_nodes in num_node_values:
            # Collect results for each scenario into lists
            all_dijkstra = [aggregated_results[num_nodes][rep]['Dijkstra'] for rep in range(repititions)]
            all_greedy = {f'Greedy-{k}': [aggregated_results[num_nodes][rep][f'Greedy-{k}'] for rep in range(repititions)] 
                        for k in range(1, k_max+1)}
            all_neighbors = {f'Avg_{k}_Hop_Neighbors': [aggregated_results[num_nodes][rep][f'Avg_{k}_Hop_Neighbors'] for rep in range(repititions)]
                            for k in range(1, k_max+1)}
            all_hop_stretch = {f'Greedy_{k}_Hop_Stretch': [aggregated_results[num_nodes][rep][f'Greedy_{k}_Hop_Stretch'] for rep in range(repititions)]
                            for k in range(1, k_max+1)}

            # Calculate means and MoEs for Dijkstra, Greedy, Neighbors, and Hop Stretch
            mean_dijkstra, _, moe_dijkstra = confidence_interval_init(all_dijkstra)
            means_greedy, moes_greedy = {}, {}
            means_neighbors, moes_neighbors = {}, {}
            means_hop_stretch, moes_hop_stretch = {}, {}

            for k in range(1, k_max+1):
                means_greedy[f'Greedy-{k}'], _, moes_greedy[f'Greedy-{k}'] = confidence_interval_init(all_greedy[f'Greedy-{k}'])
                means_neighbors[f'Avg_{k}_Hop_Neighbors'], _, moes_neighbors[f'Avg_{k}_Hop_Neighbors'] = confidence_interval_init(all_neighbors[f'Avg_{k}_Hop_Neighbors'])
                means_hop_stretch[f'Greedy_{k}_Hop_Stretch'], _, moes_hop_stretch[f'Greedy_{k}_Hop_Stretch'] = confidence_interval_init(all_hop_stretch[f'Greedy_{k}_Hop_Stretch'])

            # Write the data row
            writer.writerow({
                'NumNodes': num_nodes,
                'Avg_Dijkstra': mean_dijkstra, 'MoE_Dijkstra': moe_dijkstra,
                **{f'Avg_Greedy{k}': means_greedy[f'Greedy-{k}'] for k in range(1, k_max+1)},
                **{f'MoE_Greedy{k}': moes_greedy[f'Greedy-{k}'] for k in range(1, k_max+1)},
                **{f'Avg_{k}_Hop_Neighbors': means_neighbors[f'Avg_{k}_Hop_Neighbors'] for k in range(1, k_max+1)},
                **{f'MoE_{k}_Hop_Neighbors': moes_neighbors[f'Avg_{k}_Hop_Neighbors'] for k in range(1, k_max+1)},
                **{f'Greedy_{k}_Hop_Stretch': means_hop_stretch[f'Greedy_{k}_Hop_Stretch'] for k in range(1, k_max+1)},
                **{f'MoE_Greedy_{k}_Hop_Stretch': moes_hop_stretch[f'Greedy_{k}_Hop_Stretch'] for k in range(1, k_max+1)}
            })

    print(f"Saved final averages to {simulation_csv_path} and {simulation_average_csv_path}")