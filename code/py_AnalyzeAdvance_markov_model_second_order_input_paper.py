#!/usr/bin/env python3
"""
py_AnalyzeAdvance_markov_model_second_order_input_paper.py

Simulates geographic routing advance probabilities in an ad-hoc airborne network.
Performs:
  - Monte Carlo over node counts and repetitions.
  - Computation of overall and hop‐by‐hop advance probabilities.
  - Conditional (second‐hop) probabilities.
  - Entropy and conditional entropy.
  - Distance‐vs‐hop heatmaps.
  - Ridgeline distance distributions.
  - Topological advance plots for memoryless greedy (R=1,2,3).
  - Combined entropy‐by‐hop plots.
Outputs CSVs and PDF figures.
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

from confidence_intervals import confidence_interval_init



# -----------------------------------------------------------------------------
# Matplotlib LaTeX settings
# -----------------------------------------------------------------------------
plt.rcParams.update({
    'font.family': 'lmodern',
    'font.size': 30,
    'text.usetex': True,
    'pgf.rcfonts': False,
    'savefig.dpi': 300,
    'text.latex.preamble': r"""
        \usepackage{lmodern}
        \usepackage{amsmath}
        \usepackage{graphicx}
        \usepackage{textcomp}
        \newcommand{\pPlusTA}{p_{\scalebox{0.9}{\large TA+}}}
        \newcommand{\pZeroTA}{p_{\scalebox{0.9}{\large TA\ensuremath{\circ}}}}
        \newcommand{\pMinusTA}{p_{\scalebox{0.9}{\large TA-}}}
        \newcommand{\pFailTA}{p_{\scalebox{0.9}{\large F}}}

        \newcommand{\pPlusPlusTA}{p_{\scalebox{0.9}{\large TA+ \textbar TA+}}}
        \newcommand{\pZeroPlusTA}{p_{\scalebox{0.9}{\large TA\ensuremath{\circ} \textbar TA+}}}
        \newcommand{\pMinusPlusTA}{p_{\scalebox{0.9}{\large TA- \textbar TA+}}}
        \newcommand{\pFailPlusTA}{p_{\scalebox{0.9}{\large F \textbar TA+}}}

        \newcommand{\pPlusZeroTA}{p_{\scalebox{0.9}{\large TA+ \textbar TA\ensuremath{\circ}}}}
        \newcommand{\pZeroZeroTA}{p_{\scalebox{0.9}{\large TA\ensuremath{\circ} \textbar TA\ensuremath{\circ}}}}
        \newcommand{\pMinusZeroTA}{p_{\scalebox{0.9}{\large TA- \textbar TA\ensuremath{\circ}}}}
        \newcommand{\pFailZeroTA}{p_{\scalebox{0.9}{\large F \textbar TA\ensuremath{\circ}}}}

        \newcommand{\pPlusMinusTA}{p_{\scalebox{0.9}{\large TA+ \textbar TA-}}}
        \newcommand{\pZeroMinusTA}{p_{\scalebox{0.9}{\large TA\ensuremath{\circ} \textbar TA-}}}
        \newcommand{\pMinusMinusTA}{p_{\scalebox{0.9}{\large TA- \textbar TA-}}}
        \newcommand{\pFailMinusTA}{p_{\scalebox{0.9}{\large F \textbar TA-}}}

        \newcommand{\entropyTAPlus}{H_{\scalebox{0.9}{\Large TA+}}}
        \newcommand{\entropyTAMinus}{H_{\scalebox{0.9}{\Large TA-}}}
        \newcommand{\entropyTAZero}{H_{\scalebox{0.9}{\Large TA\ensuremath{\circ}}}}
    """
})

def create_network(area_length, area_side, num_nodes,
                   a2a_comm_range, a2g_comm_range,
                   GS_id, GS_pos):
    """
    Create a network graph of mobile nodes plus a ground station.

    Parameters:
        area_length (float): Length of the deployment area (x‑dimension).
        area_side (float):   Width of the deployment area (y‑dimension).
        num_nodes (int):     Number of mobile nodes to place.
        a2a_comm_range (float): Maximum distance for air‑to‑air links.
        a2g_comm_range (float): Maximum distance for air‑to‑ground links.
        GS_id (int):         Identifier for the ground station node.
        GS_pos (tuple):      (x, y) coordinates of the ground station.

    Returns:
        G (networkx.Graph): Graph whose nodes are 0..num_nodes-1 and GS_id,
                            with edges for node‑node and node‑GS links.
        positions (dict):    Mapping node_id → (x, y) position for all nodes,
                            including the ground station.
    """
    # 1) Randomly place mobile nodes
    positions = {
        i: (np.random.uniform(0, area_length),
            np.random.uniform(0, area_side))
        for i in range(num_nodes)
    }

    # 2) Initialize graph and add mobile nodes
    G = nx.Graph()
    G.add_nodes_from(positions.keys())

    # 3) Connect any two mobile nodes within air‑to‑air range
    for i in positions:
        for j in positions:
            if i < j:
                dx = positions[i][0] - positions[j][0]
                dy = positions[i][1] - positions[j][1]
                if np.hypot(dx, dy) <= a2a_comm_range:
                    G.add_edge(i, j)

    # 4) Add the ground station node
    positions[GS_id] = GS_pos
    G.add_node(GS_id)

    # 5) Connect each mobile node to GS if within air‑to‑ground range
    for node, pos in positions.items():
        if node == GS_id:
            continue
        dx = GS_pos[0] - pos[0]
        dy = GS_pos[1] - pos[1]
        if np.hypot(dx, dy) <= a2g_comm_range:
            G.add_edge(GS_id, node)

    return G, positions

def process_all_nodes_with_distance(G, positions, radius=7, GS_id=501):
    """
    Compute neighbor info, Euclidean distance, and hop count to a destination (GS_id) for each node.

    Inputs:
    ------
    G : networkx.Graph
        Graph containing nodes and edges.
    positions : dict[int, tuple[float, float]]
        Mapping from node ID to (x, y) coordinates, including GS_id.
    radius : int, optional
        Maximum hop‐distance to consider when discovering neighbors (default 7).
    GS_id : int, optional
        Node ID of the ground station / destination (default 501).

    Returns:
    -------
    dict[int, dict]
        Mapping each node_id (excluding GS_id) → {
            'neighbors_info': dict[str, list[tuple[int, int]]],
                Keys are f"{h}-hop" for h in 1..radius;
                Values are lists of (neighbor_node, via_node) pairs.
            'distance_to_GS': float,
                Euclidean distance from node to GS_id.
            'hop_count_to_GS': int or float('inf'),
                Graph‐distance (number of hops) to GS_id, or infinity if unreachable.
        }
    """
    # Precompute all‐pairs Euclidean distances
    precomputed = {
        (u, v): np.linalg.norm(np.subtract(positions[u], positions[v]))
        for u in G.nodes() for v in G.nodes()
    }

    results = {}
    for u in G.nodes():
        if u == GS_id:
            continue

        # Shortest paths up to `radius` hops from u
        paths = nx.single_source_shortest_path(G, u, cutoff=radius)

        # Prepare neighbor buckets for each hop‐distance
        neighbors_info = {f"{h}-hop": [] for h in range(1, radius + 1)}
        for v, path in paths.items():
            if v in (u, GS_id):
                continue
            hops = len(path) - 1
            if hops <= radius:
                # Gather all shortest paths if path length > 2
                if len(path) > 2:
                    all_paths = nx.all_shortest_paths(G, source=u, target=v)
                else:
                    all_paths = [path]
                # Choose the via-node maximizing Euclidean advance
                via = max(
                    (p[1] for p in all_paths if len(p) > 1),
                    key=lambda w: precomputed[(u, w)],
                    default=None
                )
                neighbors_info[f"{hops}-hop"].append((v, via))

        # Euclidean distance to ground station
        dist_to_gs = np.linalg.norm(np.subtract(positions[u], positions[GS_id]))
        # Hop count to ground station
        if nx.has_path(G, u, GS_id):
            hop_count = len(nx.shortest_path(G, u, GS_id)) - 1
        else:
            hop_count = float('inf')

        results[u] = {
            'neighbors_info': neighbors_info,
            'distance_to_GS': dist_to_gs,
            'hop_count_to_GS': hop_count
        }

    return results

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

def calculate_advance_probabilities(G, all_results, a2a_comm_range, a2g_comm_range, destination, find_next_hop, k):
    """
    Calculate the probabilities for positive, zero, and negative advances,
    as well as failure rates in a geographic routing scenario.

    Parameters:
    - G : networkx.Graph
        The graph over which nodes are routed.
    - all_results : dict
        A dictionary containing node-specific details like distances to GS.
    - a2a_comm_range : float
        Air-to-air communication range.
    - a2g_comm_range : float
        Air-to-ground communication range.
    - destination : int
        The destination node ID.
    - find_next_hop : callable
        Function to determine the best next hop.
    - k : int
        Maximum number of hops to consider.

    Returns:
    - tuple of floats:
        Probability of positive advance, zero advance, negative advance, and failure.
    """
    # Compute hop counts from each node to the destination using Dijkstra's algorithm
    hop_counts_to_dest = nx.single_source_shortest_path_length(G, destination)
    max_hop = max(hop_counts_to_dest.values())

    # Initialize the next hop dictionary with -1 for all nodes except the destination
    next_hop_dict = {node: -1 for node in hop_counts_to_dest if node != destination}

    # Initialize the topological_prob_k dict
    topological_prob_k = {"positive": {}, "zero": {}, "negative": {}, "failure": {}}

    # Iterate through all nodes except the destination
    for node in G.nodes:
        if node != destination and node in hop_counts_to_dest:
            distance_to_GS = all_results[node]['distance_to_GS']
            
            # Check if the node can directly connect to the destination within the A2G communication range
            if distance_to_GS <= a2g_comm_range:
                next_hop_dict[node] = destination
            else:
                # Find the best next hop for nodes not directly connecting to the destination
                next_hop = find_next_hop(node, all_results, a2a_comm_range, k)
                if next_hop:
                    next_hop_dict[node] = next_hop

    # Exclude nodes whose next hop is the destination from the calculation.
    filtered_next_hop_dict = {node: next_hop for node, next_hop in next_hop_dict.items() if next_hop != destination}
    
    # Step 1: Nodes with valid next hops (i.e. not -1) among the filtered ones.
    valid_next_hops_nodes = [node for node, next_hop in filtered_next_hop_dict.items() if next_hop != -1]

    # Step 2: Create another dictionary with hop counts of nodes and their next hops
    hop_count_advances = {}
    for node, next_hop in filtered_next_hop_dict.items():
        if next_hop != -1:  # Ensure the next hop exists
            hop_count_advances[node] = (hop_counts_to_dest[node], hop_counts_to_dest[next_hop])

    # Step 3: Calculate probabilities for positive, negative, and zero advances
    # Total nodes in next_hop_dict
    total_nodes_in_dict = len(filtered_next_hop_dict)
    positive_advances = sum(1 for counts in hop_count_advances.values() if counts[1] < counts[0])  # Positive advance
    negative_advances = sum(1 for counts in hop_count_advances.values() if counts[1] > counts[0])  # Negative advance
    zero_advances = sum(1 for counts in hop_count_advances.values() if counts[1] == counts[0])  # Zero advance
    failures = total_nodes_in_dict - len(valid_next_hops_nodes)  # Adjusted zero advance

    # Calculate probabilities dividing by total number in next_hop_dict
    prob_positive_advance = positive_advances / total_nodes_in_dict if total_nodes_in_dict > 0 else 0
    prob_negative_advance = negative_advances / total_nodes_in_dict if total_nodes_in_dict > 0 else 0
    prob_zero_advance = zero_advances / total_nodes_in_dict if total_nodes_in_dict > 0 else 0
    prob_failure = failures / total_nodes_in_dict if total_nodes_in_dict > 0 else 0
    
    # print(f"\nGreedy-{k}:")

    for hop in range(1, max_hop + 1):
        nodes_at_hop = [node for node, hop_count in hop_counts_to_dest.items() if hop_count == hop]
        total_nodes = len(nodes_at_hop)

        if total_nodes == 0:
            continue

        positive, zero, negative, failure = 0, 0, 0, 0

        for node in nodes_at_hop:
            next_hop = next_hop_dict.get(node)
            if next_hop == -1:
                # No valid next hop, categorized as failure
                failure += 1
            elif hop_counts_to_dest[next_hop] < hop_counts_to_dest[node]:
                # Positive topological advance
                positive += 1
            elif hop_counts_to_dest[next_hop] > hop_counts_to_dest[node]:
                # Negative topological advance
                negative += 1
            elif hop_counts_to_dest[next_hop] == hop_counts_to_dest[node]:
                # Zero topological advance
                zero += 1

        topological_prob_k["positive"][hop] = positive / total_nodes
        topological_prob_k["zero"][hop] = zero / total_nodes
        topological_prob_k["negative"][hop] = negative / total_nodes
        topological_prob_k["failure"][hop] = failure / total_nodes

    # --- Compute Conditional Second-Hop Probabilities ---
    # Exclude nodes whose next hop is the destination.
    filtered_next_hop_dict = {node: nh for node, nh in next_hop_dict.items() if nh != destination}
    conditional_counts = {
        'TA+': {'PTA+': 0, 'PTA0': 0, 'PTA-': 0, 'fail': 0, 'total': 0},
        'TA0': {'PTA+': 0, 'PTA0': 0, 'PTA-': 0, 'fail': 0, 'total': 0},
        'TA-': {'PTA+': 0, 'PTA0': 0, 'PTA-': 0, 'fail': 0, 'total': 0}
    }
    for node, first_hop in filtered_next_hop_dict.items():
        if first_hop == -1:
            continue
        if hop_counts_to_dest[first_hop] < hop_counts_to_dest[node]:
            first_class = 'TA+'
        elif hop_counts_to_dest[first_hop] == hop_counts_to_dest[node]:
            first_class = 'TA0'
        else:
            first_class = 'TA-'
        conditional_counts[first_class]['total'] += 1
        if first_hop == destination:
            continue
        if first_hop in next_hop_dict and next_hop_dict[first_hop] != -1:
            second_hop = next_hop_dict[first_hop]
            if hop_counts_to_dest[second_hop] < hop_counts_to_dest[first_hop]:
                second_class = 'PTA+'
            elif hop_counts_to_dest[second_hop] == hop_counts_to_dest[first_hop]:
                second_class = 'PTA0'
            else:
                second_class = 'PTA-'
            conditional_counts[first_class][second_class] += 1
        else:
            conditional_counts[first_class]['fail'] += 1

    conditional_probabilities = {}
    for first_class, counts in conditional_counts.items():
        total_count = counts['total']
        if total_count > 0:
            conditional_probabilities[first_class] = {
                'PTA+': counts['PTA+'] / total_count,
                'PTA0': counts['PTA0'] / total_count,
                'PTA-': counts['PTA-'] / total_count,
                'fail': counts['fail'] / total_count
            }
        else:
            conditional_probabilities[first_class] = {'PTA+': 0, 'PTA0': 0, 'PTA-': 0, 'fail': 0}

    # --- Calculate and Print Conditional Probabilities by Hop ---
    # print("\n####################################################\n")
    # print(f"Num Nodes: {G.number_of_nodes()-1}")
    # print("\nConditional Probabilities by Hop (Excluding hops 0 and 1):")
    conditional_probabilities_k = {}
    for hop in range(1, max_hop + 1):
        nodes_at_hop = [node for node, hop_count in hop_counts_to_dest.items() if hop_count == hop]

        conditional_counts_by_hop = {
            'TA+': {'PTA+': 0, 'PTA0': 0, 'PTA-': 0, 'fail': 0, 'total': 0},
            'TA0': {'PTA+': 0, 'PTA0': 0, 'PTA-': 0, 'fail': 0, 'total': 0},
            'TA-': {'PTA+': 0, 'PTA0': 0, 'PTA-': 0, 'fail': 0, 'total': 0}
        }

        for current_node in nodes_at_hop:
            next_hop_current = next_hop_dict.get(current_node, -1)

            if next_hop_current == -1:
                current_to_next_class = 'fail'
            else:
                if hop_counts_to_dest[next_hop_current] < hop_counts_to_dest[current_node]:
                    current_to_next_class = 'PTA+'
                elif hop_counts_to_dest[next_hop_current] == hop_counts_to_dest[current_node]:
                    current_to_next_class = 'PTA0'
                else:
                    current_to_next_class = 'PTA-'

            prev_nodes = [node for node, next_hop in next_hop_dict.items() if next_hop == current_node]
            for prev_node in prev_nodes:
                if hop_counts_to_dest[current_node] < hop_counts_to_dest[prev_node]:
                    prev_class = 'TA+'
                elif hop_counts_to_dest[current_node] == hop_counts_to_dest[prev_node]:
                    prev_class = 'TA0'
                else:
                    prev_class = 'TA-'

                conditional_counts_by_hop[prev_class]['total'] += 1
                conditional_counts_by_hop[prev_class][current_to_next_class] += 1

        conditional_probabilities_k[hop] = {}
        for prev_class, counts in conditional_counts_by_hop.items():
            total = counts['total']
            if total > 0:
                probs = {
                    'PTA+': counts['PTA+'] / total,
                    'PTA0': counts['PTA0'] / total,
                    'PTA-': counts['PTA-'] / total,
                    'fail': counts['fail'] / total
                }
                # # Adjustment for stuck state or no transitions
                # if probs['PTA0'] == 1 or all(v == 0 for v in probs.values()):
                #     probs = {'PTA+': 0, 'PTA0': 0, 'PTA-': 0, 'fail': 1}
                conditional_probabilities_k[hop][prev_class] = probs
            else:
                conditional_probabilities_k[hop][prev_class] = {'PTA+': 0, 'PTA0': 0, 'PTA-': 0, 'fail': 1}

    return (prob_positive_advance, prob_zero_advance, prob_negative_advance, prob_failure,
            topological_prob_k, conditional_probabilities, conditional_probabilities_k)

def apply_confidence_interval_conditional(group):
    """
    Compute mean and margin‑of‑error for overall and conditional advance probabilities.

    Parameters
    ----------
    group : pandas.DataFrame
        Must contain the columns:
            - 'ProbabilityPositiveAdvance', 'ProbabilityZeroAdvance',
              'ProbabilityNegativeAdvance', 'ProbabilityFailure'
            - 'P(TA+|TA+)', 'P(TA0|TA+)', 'P(TA-|TA+)', 'P(F|TA+)'
            - 'P(TA+|TA0)', 'P(TA0|TA0)', 'P(TA-|TA0)', 'P(F|TA0)'
            - 'P(TA+|TA-)', 'P(TA0|TA-)', 'P(TA-|TA-)', 'P(F|TA-)'

    Returns
    -------
    pandas.Series
        Contains:
          - 'ProbabilityPositiveAdvance', 'ProbabilityZeroAdvance',
            'ProbabilityNegativeAdvance', 'ProbabilityFailure'
          - 'ProbabilityPositiveAdvanceMoE', 'ProbabilityZeroAdvanceMoE',
            'ProbabilityNegativeAdvanceMoE', 'ProbabilityFailureMoE'
          - For each conditional P(outcome|condition) and its MoE,
            keys like 'P(TA+|TA+)','P(TA+|TA+)MoE', …, 'P(F|TA-)','P(F|TA-)MoE'.
    """
    # Overall advance probabilities
    mean_plus, _, moe_plus     = confidence_interval_init(group['ProbabilityPositiveAdvance'])
    mean_zero, _, moe_zero     = confidence_interval_init(group['ProbabilityZeroAdvance'])
    mean_minus, _, moe_minus   = confidence_interval_init(group['ProbabilityNegativeAdvance'])
    mean_fail, _, moe_fail     = confidence_interval_init(group['ProbabilityFailure'])

    # Conditional given TA+
    mean_tap_tap, _, moe_tap_tap = confidence_interval_init(group['P(TA+|TA+)'])
    mean_ta0_tap, _, moe_ta0_tap = confidence_interval_init(group['P(TA0|TA+)'])
    mean_tam_tap, _, moe_tam_tap = confidence_interval_init(group['P(TA-|TA+)'])
    mean_f_tap, _, moe_f_tap     = confidence_interval_init(group['P(F|TA+)'])

    # Conditional given TA0
    mean_tap_ta0, _, moe_tap_ta0 = confidence_interval_init(group['P(TA+|TA0)'])
    mean_ta0_ta0, _, moe_ta0_ta0 = confidence_interval_init(group['P(TA0|TA0)'])
    mean_tam_ta0, _, moe_tam_ta0 = confidence_interval_init(group['P(TA-|TA0)'])
    mean_f_ta0, _, moe_f_ta0     = confidence_interval_init(group['P(F|TA0)'])

    # Conditional given TA-
    mean_tap_tam, _, moe_tap_tam = confidence_interval_init(group['P(TA+|TA-)'])
    mean_ta0_tam, _, moe_ta0_tam = confidence_interval_init(group['P(TA0|TA-)'])
    mean_tam_tam, _, moe_tam_tam = confidence_interval_init(group['P(TA-|TA-)'])
    mean_f_tam, _, moe_f_tam     = confidence_interval_init(group['P(F|TA-)'])

    return pd.Series({
        # overall
        'ProbabilityPositiveAdvance':    mean_plus,
        'ProbabilityZeroAdvance':        mean_zero,
        'ProbabilityNegativeAdvance':    mean_minus,
        'ProbabilityFailure':            mean_fail,
        'ProbabilityPositiveAdvanceMoE': moe_plus,
        'ProbabilityZeroAdvanceMoE':     moe_zero,
        'ProbabilityNegativeAdvanceMoE': moe_minus,
        'ProbabilityFailureMoE':         moe_fail,

        # P(...|TA+)
        'P(TA+|TA+)':    mean_tap_tap,
        'P(TA+|TA+)MoE': moe_tap_tap,
        'P(TA0|TA+)':    mean_ta0_tap,
        'P(TA0|TA+)MoE': moe_ta0_tap,
        'P(TA-|TA+)':    mean_tam_tap,
        'P(TA-|TA+)MoE': moe_tam_tap,
        'P(F|TA+)':      mean_f_tap,
        'P(F|TA+)MoE':   moe_f_tap,

        # P(...|TA0)
        'P(TA+|TA0)':    mean_tap_ta0,
        'P(TA+|TA0)MoE': moe_tap_ta0,
        'P(TA0|TA0)':    mean_ta0_ta0,
        'P(TA0|TA0)MoE': moe_ta0_ta0,
        'P(TA-|TA0)':    mean_tam_ta0,
        'P(TA-|TA0)MoE': moe_tam_ta0,
        'P(F|TA0)':      mean_f_ta0,
        'P(F|TA0)MoE':   moe_f_ta0,

        # P(...|TA-)
        'P(TA+|TA-)':    mean_tap_tam,
        'P(TA+|TA-)MoE': moe_tap_tam,
        'P(TA0|TA-)':    mean_ta0_tam,
        'P(TA0|TA-)MoE': moe_ta0_tam,
        'P(TA-|TA-)':    mean_tam_tam,
        'P(TA-|TA-)MoE': moe_tam_tam,
        'P(F|TA-)':      mean_f_tam,
        'P(F|TA-)MoE':   moe_f_tam,
    })

def apply_entropy_confidence_interval(group):
    """
    Compute mean and margin‑of‑error for entropy metrics in a grouped DataFrame.

    Parameters
    ----------
    group : pandas.DataFrame
        Must contain the columns:
          - 'Entropy'
          - 'Entropy_plus'
          - 'Entropy_zero'
          - 'Entropy_minus'
          - 'Entropy_fail'

    Returns
    -------
    pandas.Series
        A Series with the following entries:
          - 'Entropy_mean',      'Entropy_MoE'
          - 'Entropy_plus_mean', 'Entropy_plus_MoE'
          - 'Entropy_zero_mean', 'Entropy_zero_MoE'
          - 'Entropy_minus_mean','Entropy_minus_MoE'
          - 'Entropy_fail_mean', 'Entropy_fail_MoE'
    """
    # Overall entropy
    entropy_mean, _, entropy_moe = confidence_interval_init(group['Entropy'])
    # Per‑outcome entropies
    entropy_plus_mean, _, entropy_plus_moe   = confidence_interval_init(group['Entropy_plus'])
    entropy_zero_mean, _, entropy_zero_moe   = confidence_interval_init(group['Entropy_zero'])
    entropy_minus_mean, _, entropy_minus_moe = confidence_interval_init(group['Entropy_minus'])
    entropy_fail_mean, _, entropy_fail_moe   = confidence_interval_init(group['Entropy_fail'])

    return pd.Series({
        'Entropy_mean':        entropy_mean,
        'Entropy_MoE':         entropy_moe,
        'Entropy_plus_mean':   entropy_plus_mean,
        'Entropy_plus_MoE':    entropy_plus_moe,
        'Entropy_zero_mean':   entropy_zero_mean,
        'Entropy_zero_MoE':    entropy_zero_moe,
        'Entropy_minus_mean':  entropy_minus_mean,
        'Entropy_minus_MoE':   entropy_minus_moe,
        'Entropy_fail_mean':   entropy_fail_mean,
        'Entropy_fail_MoE':    entropy_fail_moe
    })

def apply_entropy_confidence_interval_conditional(group):
    """
    Compute mean and margin‑of‑error for conditional entropy metrics in a grouped DataFrame.

    Parameters
    ----------
    group : pandas.DataFrame
        Must contain the columns:
          - 'Entropy_cond'
          - 'Entropy_cond_plus'
          - 'Entropy_cond_zero'
          - 'Entropy_cond_minus'

    Returns
    -------
    pandas.Series
        A Series with the following entries:
          - 'Entropy_cond_mean',       'Entropy_cond_MoE'
          - 'Entropy_cond_plus_mean',  'Entropy_cond_plus_MoE'
          - 'Entropy_cond_zero_mean',  'Entropy_cond_zero_MoE'
          - 'Entropy_cond_minus_mean', 'Entropy_cond_minus_MoE'
    """
    # Overall conditional entropy
    cond_mean, _, cond_moe = confidence_interval_init(group['Entropy_cond'])
    # Per‑state conditional entropies
    cond_plus_mean, _, cond_plus_moe   = confidence_interval_init(group['Entropy_cond_plus'])
    cond_zero_mean, _, cond_zero_moe   = confidence_interval_init(group['Entropy_cond_zero'])
    cond_minus_mean, _, cond_minus_moe = confidence_interval_init(group['Entropy_cond_minus'])

    return pd.Series({
        'Entropy_cond_mean':        cond_mean,
        'Entropy_cond_MoE':         cond_moe,
        'Entropy_cond_plus_mean':   cond_plus_mean,
        'Entropy_cond_plus_MoE':    cond_plus_moe,
        'Entropy_cond_zero_mean':   cond_zero_mean,
        'Entropy_cond_zero_MoE':    cond_zero_moe,
        'Entropy_cond_minus_mean':  cond_minus_mean,
        'Entropy_cond_minus_MoE':   cond_minus_moe
    })

def apply_confidence_interval(group):
    """
    Compute mean and margin‑of‑error for overall advance probabilities.

    Parameters
    ----------
    group : pandas.DataFrame
        Must contain the columns:
          - 'ProbabilityPositiveAdvance'
          - 'ProbabilityNegativeAdvance'
          - 'ProbabilityZeroAdvance'
          - 'ProbabilityFailure'

    Returns
    -------
    pandas.Series
        A Series with the following entries:
          - 'ProbabilityPositiveAdvance', 'ProbabilityPositiveAdvanceMoE'
          - 'ProbabilityNegativeAdvance', 'ProbabilityNegativeAdvanceMoE'
          - 'ProbabilityZeroAdvance', 'ProbabilityZeroAdvanceMoE'
          - 'ProbabilityFailure', 'ProbabilityFailureMoE'
    """
    # Positive advance
    mean_plus, _, moe_plus = confidence_interval_init(group['ProbabilityPositiveAdvance'])
    # Negative advance
    mean_minus, _, moe_minus = confidence_interval_init(group['ProbabilityNegativeAdvance'])
    # Zero advance
    mean_zero, _, moe_zero = confidence_interval_init(group['ProbabilityZeroAdvance'])
    # Failure
    mean_failure, _, moe_failure = confidence_interval_init(group['ProbabilityFailure'])

    return pd.Series({
        'ProbabilityPositiveAdvance':    mean_plus,
        'ProbabilityNegativeAdvance':    mean_minus,
        'ProbabilityZeroAdvance':        mean_zero,
        'ProbabilityFailure':            mean_failure,
        'ProbabilityPositiveAdvanceMoE': moe_plus,
        'ProbabilityNegativeAdvanceMoE': moe_minus,
        'ProbabilityZeroAdvanceMoE':     moe_zero,
        'ProbabilityFailureMoE':         moe_failure
    })

def entropy_contribution(p, num_outcomes=4):
    """
    Compute the normalized entropy contribution of a single probability.

    Parameters
    ----------
    p : float
        Probability of one outcome (0 ≤ p ≤ 1).
    num_outcomes : int, optional
        Total number of possible outcomes for normalization (default is 4).

    Returns
    -------
    float
        −p * log2(p) / log2(num_outcomes) if p > 0, otherwise 0.
        This yields a value between 0 and 1, where 1 corresponds to maximum entropy
        when all outcomes are equally likely (p = 1/num_outcomes).
    """
    if p <= 0:
        return 0.0
    return -p * np.log2(p) / np.log2(num_outcomes)

def add_overall_conditional_entropy(df):
    """
    Compute and add overall conditional entropy column 'Entropy_cond'.

    Assumes df contains:
        'Entropy_plus', 'Entropy_zero', 'Entropy_minus',
        'Entropy_cond_plus', 'Entropy_cond_zero', 'Entropy_cond_minus'

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    pandas.DataFrame
        Input df with new column 'Entropy_cond'.
    """
    df['Entropy_cond'] = (
        df['Entropy_plus']   * df['Entropy_cond_plus'] +
        df['Entropy_zero']   * df['Entropy_cond_zero'] +
        df['Entropy_minus']  * df['Entropy_cond_minus']
    )
    return df
    
def run_simulation(args):
    """
    Run a single simulation replicate.

    Inputs:
        args (tuple): (num_nodes, rep, k_max)
    Returns:
        list: [
            num_nodes, rep,
            probability_positive_advance (dict),
            probability_zero_advance (dict),
            probability_negative_advance (dict),
            probability_failure (dict),
            topological_probabilities (dict),
            conditional_probabilities (dict),
            conditional_probabilities_by_hops (dict),
            filtered_nodes (dict)
        ]
    """
    num_nodes, rep, k_max = args
    np.random.seed(rep)

    # Communication ranges and area dimensions
    a2a_comm_range = 100
    a2g_comm_range = 370.4
    area_length, area_side = 1250, 800
    GS_id, GS_pos = 501, (1150, 400)

    # Build network and process nodes
    G, positions = create_network(
        area_length, area_side, num_nodes,
        a2a_comm_range, a2g_comm_range,
        GS_id, GS_pos
    )
    all_node_details = process_all_nodes_with_distance(
        G, positions, radius=k_max, GS_id=GS_id
    )

    # Keep only nodes that can reach GS
    filtered_nodes = {
        node: details
        for node, details in all_node_details.items()
        if details['hop_count_to_GS'] != float('inf')
    }

    # Initialize result containers
    prob_pos = {}
    prob_zero = {}
    prob_neg = {}
    prob_fail = {}
    topo_probs = {}
    cond_probs = {}
    cond_probs_by_hops = {}

    for k in range(1, k_max + 1):
        p_pos, p_zero, p_neg, p_fail, topo_k, cond_k, cond_hops_k = calculate_advance_probabilities(
            G, all_node_details,
            a2a_comm_range, a2g_comm_range,
            GS_id, best_advance_within_k_hops, k
        )
        prob_pos[k] = p_pos
        prob_zero[k] = p_zero
        prob_neg[k] = p_neg
        prob_fail[k] = p_fail
        topo_probs[k] = topo_k
        cond_probs[k] = cond_k
        cond_probs_by_hops[k] = cond_hops_k

    # Log progress
    print(
        f"Completed: NumNodes={num_nodes}, Rep={rep}, " +
        ", ".join(f"k={k}:{round(prob_pos[k],3)}" for k in prob_pos)
    )

    return [
        num_nodes, rep,
        prob_pos, prob_zero, prob_neg, prob_fail,
        topo_probs, cond_probs, cond_probs_by_hops,
        filtered_nodes
    ]


if __name__ == "__main__":
    # Communication & simulation parameters
    # repititions     = 10_000  # number of Monte‑Carlo runs used in the paper (HPC can easily handle this)
    repititions     = 5        # for quick local testing; paper results used 10,000 runs
    k_max           = 3
    num_node_values = np.arange(50, 501, 50)
    max_num_nodes   = 500
    equipage_fraction_values = num_node_values / max_num_nodes

    # —————————————————————————————————————————————
    # 2) Directory setup
    # —————————————————————————————————————————————
    # 1. Find the directory this script lives in
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 2. Go one level up to the project root
    project_root = os.path.abspath(os.path.join(script_dir, os.pardir))

    # 3. Now build your results folder there
    base_csv_dir = os.path.join(project_root, "results", "csv_files")

    dir_advance  = os.path.join(base_csv_dir, "AnalyzeAdvanceMarkov")
    dir_hops     = os.path.join(base_csv_dir, "AnalyzeAdvanceMarkov_By_Hops")
    os.makedirs(dir_advance, exist_ok=True)
    os.makedirs(dir_hops,    exist_ok=True)

    base_plot_dir = os.path.join(project_root, "results", "plots")
    plot_topo_dir     = os.path.join(base_plot_dir, "topological_advance")
    plot_entropy_dir  = os.path.join(base_plot_dir, "entropy")
    plot_distance_dir = os.path.join(base_plot_dir, "distance_vs_hops")
    os.makedirs(plot_topo_dir,     exist_ok=True)
    os.makedirs(plot_entropy_dir,  exist_ok=True)
    os.makedirs(plot_distance_dir, exist_ok=True)

    # —————————————————————————————————————————————
    # 3) CSV filenames & paths
    # —————————————————————————————————————————————
    advance_conditional_csv_filename         = f"ProbabilityAdvanceConditional_{max_num_nodes}.csv"
    advance_conditional_by_hops_csv_filename = f"ProbabilityAdvance_by_hops_{max_num_nodes}.csv"
    distance_vs_hops_csv_filename            = f"DistanceVsHops_{max_num_nodes}.csv"
    grouped_avg_advance_filename             = f"GroupedAverage_AnalyzeAdvance_nodes_{max_num_nodes}.csv"
    grouped_avg_advance_hops_filename        = f"GroupedAverage_AnalyzeAdvance_nodes_{max_num_nodes}_hops.csv"
    grouped_avg_entropy_filename             = f"GroupedAverage_AnalyzeAdvance_nodes_{max_num_nodes}_entropy.csv"
    grouped_avg_entropy_hops_filename        = f"GroupedAverage_AnalyzeAdvance_nodes_{max_num_nodes}_entropy_hops.csv"
    grouped_avg_conditional_entropy_filename = f"GroupedAverage_AnalyzeAdvance_nodes_{max_num_nodes}_conditional_entropy.csv"

    advance_conditional_path            = os.path.join(dir_advance, advance_conditional_csv_filename)
    advance_conditional_by_hops_path    = os.path.join(dir_hops,    advance_conditional_by_hops_csv_filename)
    distance_vs_hops_csv_path           = os.path.join(dir_hops,    distance_vs_hops_csv_filename)
    grouped_avg_advance_path            = os.path.join(dir_advance, grouped_avg_advance_filename)
    grouped_avg_advance_hops_path       = os.path.join(dir_hops,    grouped_avg_advance_hops_filename)
    grouped_avg_entropy_path            = os.path.join(dir_advance, grouped_avg_entropy_filename)
    grouped_avg_entropy_hops_path       = os.path.join(dir_hops,    grouped_avg_entropy_hops_filename)
    grouped_avg_conditional_entropy_path= os.path.join(dir_advance, grouped_avg_conditional_entropy_filename)

    # —————————————————————————————————————————————
    # 5) Run simulations and write CSVs
    # —————————————————————————————————————————————
    parameters = list(product(num_node_values, range(repititions), [k_max]))
    ncpus = int(os.environ.get('SLURM_CPUS_PER_TASK', cpu_count()))
    print(f"cpus_per_task: {ncpus}")

    # —————————————————————————————————————————————
    # 4) Fieldnames
    # —————————————————————————————————————————————
    fieldnames_overall = [
        "NumNodes","Rep","Radius",
        "ProbabilityPositiveAdvance","ProbabilityZeroAdvance",
        "ProbabilityNegativeAdvance","ProbabilityFailure",
        "P(TA+|TA+)","P(TA0|TA+)","P(TA-|TA+)","P(F|TA+)",
        "P(TA+|TA0)","P(TA0|TA0)","P(TA-|TA0)","P(F|TA0)",
        "P(TA+|TA-)","P(TA0|TA-)","P(TA-|TA-)","P(F|TA-)"
    ]
    fieldnames_by_hops = ["NumNodes","Rep","Radius","Hop"] + fieldnames_overall[3:]
    fieldnames_distance = ["NumNodes","Rep","NodeId","HopCount","DistanceToGs"]

    # Run simulations and write raw CSVs
    with open(advance_conditional_path,         "w", newline="") as f_ov, \
         open(advance_conditional_by_hops_path, "w", newline="") as f_hp, \
         open(distance_vs_hops_csv_path,        "w", newline="") as f_ds:

        w_ov = csv.DictWriter(f_ov, fieldnames=fieldnames_overall);         w_ov.writeheader()
        w_hp = csv.DictWriter(f_hp, fieldnames=fieldnames_by_hops);         w_hp.writeheader()
        w_ds = csv.DictWriter(f_ds, fieldnames=fieldnames_distance);        w_ds.writeheader()

        with Pool(ncpus) as pool:
            for (num_nodes, rep,
                 prob_pos, prob_zero, prob_neg, prob_fail,
                 topo_probs, cond_probs, cond_probs_by_hops,
                 filtered_nodes) in pool.imap(run_simulation, parameters):

                # overall
                for k in prob_pos:
                    row = {
                        "NumNodes": num_nodes, "Rep": rep, "Radius": k,
                        "ProbabilityPositiveAdvance": prob_pos[k],
                        "ProbabilityZeroAdvance":     prob_zero[k],
                        "ProbabilityNegativeAdvance": prob_neg[k],
                        "ProbabilityFailure":         prob_fail[k]
                    }
                    cp = cond_probs.get(k, {})
                    row.update({
                        "P(TA+|TA+)": cp.get("TA+", {}).get("PTA+",0),
                        "P(TA0|TA+)": cp.get("TA+", {}).get("PTA0",0),
                        "P(TA-|TA+)": cp.get("TA+", {}).get("PTA-",0),
                        "P(F|TA+)":   cp.get("TA+", {}).get("fail",0),
                        "P(TA+|TA0)": cp.get("TA0", {}).get("PTA+",0),
                        "P(TA0|TA0)": cp.get("TA0", {}).get("PTA0",0),
                        "P(TA-|TA0)": cp.get("TA0", {}).get("PTA-",0),
                        "P(F|TA0)":   cp.get("TA0", {}).get("fail",0),
                        "P(TA+|TA-)": cp.get("TA-", {}).get("PTA+",0),
                        "P(TA0|TA-)": cp.get("TA-", {}).get("PTA0",0),
                        "P(TA-|TA-)": cp.get("TA-", {}).get("PTA-",0),
                        "P(F|TA-)":   cp.get("TA-", {}).get("fail",0)
                    })
                    w_ov.writerow(row)

                # by‑hops
                for k, topo in topo_probs.items():
                    for hop in topo["positive"]:
                        ch = cond_probs_by_hops.get(k, {}).get(hop, {})
                        row = {
                            "NumNodes": num_nodes, "Rep": rep, "Radius": k, "Hop": hop,
                            "ProbabilityPositiveAdvance": topo["positive"].get(hop,0),
                            "ProbabilityZeroAdvance":     topo["zero"].get(hop,0),
                            "ProbabilityNegativeAdvance": topo["negative"].get(hop,0),
                            "ProbabilityFailure":         topo["failure"].get(hop,0),
                            "P(TA+|TA+)": ch.get("TA+", {}).get("PTA+",0),
                            "P(TA0|TA+)": ch.get("TA+", {}).get("PTA0",0),
                            "P(TA-|TA+)": ch.get("TA+", {}).get("PTA-",0),
                            "P(F|TA+)":   ch.get("TA+", {}).get("fail",0),
                            "P(TA+|TA0)": ch.get("TA0", {}).get("PTA+",0),
                            "P(TA0|TA0)": ch.get("TA0", {}).get("PTA0",0),
                            "P(TA-|TA0)": ch.get("TA0", {}).get("PTA-",0),
                            "P(F|TA0)":   ch.get("TA0", {}).get("fail",0),
                            "P(TA+|TA-)": ch.get("TA-", {}).get("PTA+",0),
                            "P(TA0|TA-)": ch.get("TA-", {}).get("PTA0",0),
                            "P(TA-|TA-)": ch.get("TA-", {}).get("PTA-",0),
                            "P(F|TA-)":   ch.get("TA-", {}).get("fail",0)
                        }
                        w_hp.writerow(row)

                # distances
                for node_id, det in filtered_nodes.items():
                    w_ds.writerow({
                        "NumNodes":    num_nodes,
                        "Rep":         rep,
                        "NodeId":      node_id,
                        "HopCount":    det["hop_count_to_GS"],
                        "DistanceToGs":det["distance_to_GS"]
                    })

                print(f"Processed simulation for rep: {rep} with {num_nodes} nodes.")
    print(f"Saved final averages to {grouped_avg_advance_path}")


    # —————————————————————————————————————————————
    # 6) Post‑processing & grouping
    # —————————————————————————————————————————————

    # Read the raw CSV outputs
    probability_advance_df          = pd.read_csv(advance_conditional_path)
    probability_advance_by_hops_df = pd.read_csv(advance_conditional_by_hops_path)
    distance_vs_hops_df             = pd.read_csv(distance_vs_hops_csv_path)

    # Fix any rows where all four advance probabilities are zero → set failure=1
    mask = (
        (probability_advance_df['ProbabilityPositiveAdvance'] == 0) &
        (probability_advance_df['ProbabilityNegativeAdvance'] == 0) &
        (probability_advance_df['ProbabilityZeroAdvance']     == 0) &
        (probability_advance_df['ProbabilityFailure']         == 0)
    )
    probability_advance_df.loc[mask, 'ProbabilityFailure'] = 1

    mask = (
        (probability_advance_by_hops_df['ProbabilityPositiveAdvance'] == 0) &
        (probability_advance_by_hops_df['ProbabilityNegativeAdvance'] == 0) &
        (probability_advance_by_hops_df['ProbabilityZeroAdvance']     == 0) &
        (probability_advance_by_hops_df['ProbabilityFailure']         == 0)
    )
    probability_advance_by_hops_df.loc[mask, 'ProbabilityFailure'] = 1

    # Compute normalized entropy contributions for each outcome
    for df in (probability_advance_df, probability_advance_by_hops_df):
        df['Entropy_plus']  = df['ProbabilityPositiveAdvance'].apply(entropy_contribution)
        df['Entropy_zero']  = df['ProbabilityZeroAdvance'].apply(entropy_contribution)
        df['Entropy_minus'] = df['ProbabilityNegativeAdvance'].apply(entropy_contribution)
        df['Entropy_fail']  = df['ProbabilityFailure'].apply(entropy_contribution)
        df['Entropy']       = (
            df['Entropy_plus'] +
            df['Entropy_zero'] +
            df['Entropy_minus'] +
            df['Entropy_fail']
        )

    # — TA+ condition (overall) —
    mask = (
        (probability_advance_df['P(TA+|TA+)'] == 0) &
        (probability_advance_df['P(TA0|TA+)'] == 0) &
        (probability_advance_df['P(TA-|TA+)'] == 0) &
        (probability_advance_df['P(F|TA+)']   == 0)
    )
    probability_advance_df.loc[mask, 'P(F|TA+)'] = 1
    probability_advance_df['Entropy_cond_plus_plus'] = probability_advance_df['P(TA+|TA+)'].apply(entropy_contribution)
    probability_advance_df['Entropy_cond_plus_zero'] = probability_advance_df['P(TA0|TA+)'].apply(entropy_contribution)
    probability_advance_df['Entropy_cond_plus_minus']= probability_advance_df['P(TA-|TA+)'].apply(entropy_contribution)
    probability_advance_df['Entropy_cond_plus_fail'] = probability_advance_df['P(F|TA+)'].apply(entropy_contribution)
    probability_advance_df['Entropy_cond_plus']      = (
        probability_advance_df['Entropy_cond_plus_plus'] +
        probability_advance_df['Entropy_cond_plus_zero'] +
        probability_advance_df['Entropy_cond_plus_minus']+
        probability_advance_df['Entropy_cond_plus_fail']
    )

    # — TA0 condition (overall) —
    mask = (
        (probability_advance_df['P(TA+|TA0)'] == 0) &
        (probability_advance_df['P(TA0|TA0)'] == 0) &
        (probability_advance_df['P(TA-|TA0)'] == 0) &
        (probability_advance_df['P(F|TA0)']   == 0)
    )
    probability_advance_df.loc[mask, 'P(F|TA0)'] = 1
    probability_advance_df['Entropy_cond_zero_plus'] = probability_advance_df['P(TA+|TA0)'].apply(entropy_contribution)
    probability_advance_df['Entropy_cond_zero_zero'] = probability_advance_df['P(TA0|TA0)'].apply(entropy_contribution)
    probability_advance_df['Entropy_cond_zero_minus']= probability_advance_df['P(TA-|TA0)'].apply(entropy_contribution)
    probability_advance_df['Entropy_cond_zero_fail'] = probability_advance_df['P(F|TA0)'].apply(entropy_contribution)
    probability_advance_df['Entropy_cond_zero']      = (
        probability_advance_df['Entropy_cond_zero_plus'] +
        probability_advance_df['Entropy_cond_zero_zero'] +
        probability_advance_df['Entropy_cond_zero_minus']+
        probability_advance_df['Entropy_cond_zero_fail']
    )

    # — TA- condition (overall) —
    mask = (
        (probability_advance_df['P(TA+|TA-)'] == 0) &
        (probability_advance_df['P(TA0|TA-)'] == 0) &
        (probability_advance_df['P(TA-|TA-)'] == 0) &
        (probability_advance_df['P(F|TA-)']   == 0)
    )
    probability_advance_df.loc[mask, 'P(F|TA-)'] = 1
    probability_advance_df['Entropy_cond_minus_plus'] = probability_advance_df['P(TA+|TA-)'].apply(entropy_contribution)
    probability_advance_df['Entropy_cond_minus_zero'] = probability_advance_df['P(TA0|TA-)'].apply(entropy_contribution)
    probability_advance_df['Entropy_cond_minus_minus']= probability_advance_df['P(TA-|TA-)'].apply(entropy_contribution)
    probability_advance_df['Entropy_cond_minus_fail'] = probability_advance_df['P(F|TA-)'].apply(entropy_contribution)
    probability_advance_df['Entropy_cond_minus']      = (
        probability_advance_df['Entropy_cond_minus_plus'] +
        probability_advance_df['Entropy_cond_minus_zero'] +
        probability_advance_df['Entropy_cond_minus_minus']+
        probability_advance_df['Entropy_cond_minus_fail']
    )

    # — Repeat conditional entropy for by‑hops DataFrame —
    for df in (probability_advance_by_hops_df,):
        # TA+ by‑hops
        mask = (
            (df['P(TA+|TA+)'] == 0) &
            (df['P(TA0|TA+)'] == 0) &
            (df['P(TA-|TA+)'] == 0) &
            (df['P(F|TA+)']   == 0)
        )
        df.loc[mask, 'P(F|TA+)'] = 1
        df['Entropy_cond_plus_plus'] = df['P(TA+|TA+)'].apply(entropy_contribution)
        df['Entropy_cond_plus_zero'] = df['P(TA0|TA+)'].apply(entropy_contribution)
        df['Entropy_cond_plus_minus']= df['P(TA-|TA+)'].apply(entropy_contribution)
        df['Entropy_cond_plus_fail'] = df['P(F|TA+)'].apply(entropy_contribution)
        df['Entropy_cond_plus']      = (
            df['Entropy_cond_plus_plus'] +
            df['Entropy_cond_plus_zero'] +
            df['Entropy_cond_plus_minus']+
            df['Entropy_cond_plus_fail']
        )
        # TA0 by‑hops
        mask = (
            (df['P(TA+|TA0)'] == 0) &
            (df['P(TA0|TA0)'] == 0) &
            (df['P(TA-|TA0)'] == 0) &
            (df['P(F|TA0)']   == 0)
        )
        df.loc[mask, 'P(F|TA0)'] = 1
        df['Entropy_cond_zero_plus'] = df['P(TA+|TA0)'].apply(entropy_contribution)
        df['Entropy_cond_zero_zero'] = df['P(TA0|TA0)'].apply(entropy_contribution)
        df['Entropy_cond_zero_minus']= df['P(TA-|TA0)'].apply(entropy_contribution)
        df['Entropy_cond_zero_fail'] = df['P(F|TA0)'].apply(entropy_contribution)
        df['Entropy_cond_zero']      = (
            df['Entropy_cond_zero_plus'] +
            df['Entropy_cond_zero_zero'] +
            df['Entropy_cond_zero_minus']+
            df['Entropy_cond_zero_fail']
        )
        # TA- by‑hops
        mask = (
            (df['P(TA+|TA-)'] == 0) &
            (df['P(TA0|TA-)'] == 0) &
            (df['P(TA-|TA-)'] == 0) &
            (df['P(F|TA-)']   == 0)
        )
        df.loc[mask, 'P(F|TA-)'] = 1
        df['Entropy_cond_minus_plus'] = df['P(TA+|TA-)'].apply(entropy_contribution)
        df['Entropy_cond_minus_zero'] = df['P(TA0|TA-)'].apply(entropy_contribution)
        df['Entropy_cond_minus_minus']= df['P(TA-|TA-)'].apply(entropy_contribution)
        df['Entropy_cond_minus_fail'] = df['P(F|TA-)'].apply(entropy_contribution)
        df['Entropy_cond_minus']      = (
            df['Entropy_cond_minus_plus'] +
            df['Entropy_cond_minus_zero'] +
            df['Entropy_cond_minus_minus']+
            df['Entropy_cond_minus_fail']
        )

    # Add overall conditional entropy
    probability_advance_df          = add_overall_conditional_entropy(probability_advance_df)
    probability_advance_by_hops_df = add_overall_conditional_entropy(probability_advance_by_hops_df)

    # Group overall probabilities with confidence intervals
    grouped_average_df = (
        probability_advance_df
        .groupby(['NumNodes', 'Radius'])
        .apply(apply_confidence_interval_conditional, include_groups=False)
        .reset_index()
    )
    grouped_average_df['EquipageFraction'] = grouped_average_df['NumNodes'] / max_num_nodes
    grouped_average_df.to_csv(grouped_avg_advance_path, index=False)

    # Group by‑hops probabilities with confidence intervals
    grouped_average_hops_df = (
        probability_advance_by_hops_df
        .groupby(['NumNodes', 'Radius', 'Hop'])
        .filter(lambda g: len(g) >= 0.5 * repititions)
        .groupby(['NumNodes', 'Radius', 'Hop'])
        .apply(apply_confidence_interval_conditional, include_groups=False)
        .reset_index()
    )
    grouped_average_hops_df.to_csv(grouped_avg_advance_hops_path, index=False)

    # Group overall entropy with confidence intervals
    grouped_entropy_df = (
        probability_advance_df
        .groupby(['NumNodes', 'Radius'])
        .apply(apply_entropy_confidence_interval, include_groups=False)
        .reset_index()
    )
    grouped_entropy_df['EquipageFraction'] = grouped_entropy_df['NumNodes'] / max_num_nodes
    grouped_entropy_df.to_csv(grouped_avg_entropy_path, index=False)

    # Group by‑hops entropy with confidence intervals
    grouped_entropy_hops_df = (
        probability_advance_by_hops_df
        .groupby(['NumNodes', 'Radius', 'Hop'])
        .filter(lambda g: len(g) >= 0.5 * repititions)
        .groupby(['NumNodes', 'Radius', 'Hop'])
        .apply(apply_entropy_confidence_interval, include_groups=False)
        .reset_index()
    )
    grouped_entropy_hops_df.to_csv(grouped_avg_entropy_hops_path, index=False)

    # Group conditional entropy with confidence intervals
    grouped_cond_entropy_df = (
        probability_advance_df
        .groupby(['NumNodes', 'Radius'])
        .apply(apply_entropy_confidence_interval_conditional, include_groups=False)
        .reset_index()
    )
    grouped_cond_entropy_df['EquipageFraction'] = grouped_cond_entropy_df['NumNodes'] / max_num_nodes
    grouped_cond_entropy_df.to_csv(grouped_avg_conditional_entropy_path, index=False)

    # Group by‑hops conditional entropy with confidence intervals
    grouped_cond_entropy_hops_df = (
        probability_advance_by_hops_df
        .groupby(['NumNodes', 'Radius', 'Hop'])
        .filter(lambda g: len(g) >= 0.5 * repititions)
        .groupby(['NumNodes', 'Radius', 'Hop'])
        .apply(apply_entropy_confidence_interval_conditional, include_groups=False)
        .reset_index()
    )
    grouped_cond_entropy_hops_df.to_csv(
        os.path.join(dir_hops, f"GroupedAverage_AnalyzeAdvance_nodes_{max_num_nodes}_conditional_entropy_hops.csv"),
        index=False
    )

    # Plotting results
    plt.figure(figsize=(12, 9))
    capsize = 4
    markersize = 9
    lw = 4
    colors_map = [plt.cm.jet(x) for x in np.linspace(0, 1, 8)]  # Generate colors for each plot line

    plot_styles = [
        {'fmt': 'o', 'color': colors_map[0], 'label': 'Greedy-1'},
        {'fmt': 'o', 'color': colors_map[1], 'label': 'Greedy-2'},
        {'fmt': 'o', 'color': colors_map[2], 'label': 'Greedy-3'},
        {'fmt': 'o', 'color': colors_map[3], 'label': 'Greedy-4'},
        {'fmt': 'o', 'color': colors_map[4], 'label': 'Greedy-5'},
        {'fmt': 'o', 'color': colors_map[5], 'label': 'Greedy-6'},
        {'fmt': 'o', 'color': colors_map[6], 'label': 'Greedy-7'}
    ]

    # Filter plot_styles dynamically based on k_max
    plot_styles = plot_styles[:k_max]

    # Define a list of markers for different convex hops
    markers = ['o', 's', '^', 'v', 'D', '*', 'x', 'P']  # Add more markers if needed

    # Specify the number of nodes to filter
    # selected_num_nodes = 250  # Replace with your desired number of nodes
    selected_num_nodes = int(0.8 * max_num_nodes)  # Replace with your desired number of nodes

    # Filter the DataFrame by the selected number of nodes
    filtered_data = probability_advance_df[probability_advance_df["NumNodes"] == selected_num_nodes]

    # Group data by Radius
    radii = filtered_data["Radius"].unique()

    # Define a color palette for the radii (you can expand this for more radii)
    # radius_colors = {
    #     1: 'b',   # Blue
    #     2: 'g',   # Green
    #     3: 'r',   # Red
    #     4: 'c',   # Cyan
    #     5: 'm',   # Magenta
    #     6: 'y',   # Yellow
    #     7: 'k',   # Black
    # }
    radius_colors = {
        1: 'k',   # Black
        2: '#298C8C',   # Cyan
        # 3: 'dimgrey',   # Red
        3: 'olive',   # Red
        4: '#A00000',   # Red
    }
    
    radius_markers = {
        1: 'o',   # Black
        2: 's',   # Cyan
        3: '^',   # Red
        4: 'v',   # Red
    }

    # Define line styles and markers for each hop depth (for conditional probabilities)
    line_styles = {1: 'dashed', 2: 'dotted', 3: 'dashdot'}
    markers = {0: 'o', 1: 's', 2: '^', 3: 'v'}

    # Setup
    selected_nodes_list = [400, 200]
    rho_labels = [r'(a) $\rho = 0.8$', r'(b) $\rho = 0.4$']
    cmap_color = 'hot_r'      # ← change here

    # 1) global maxima:
    x_max_global = distance_vs_hops_df["DistanceToGs"].max()
    y_max_global = distance_vs_hops_df["HopCount"].max()

    # 2) x-bins every 20 km (for example; adjust if you really wanted 10):
    bins_x = np.arange(-10.6, x_max_global + 10.4, 20)

    # 3) y-bins of width 0.2 centered on each integer 1,2,…,max_hop:
    max_hop = int(y_max_global)
    bins_y = np.concatenate([[i - 0.15, i + 0.15] for i in range(1, max_hop + 1)])
    bins_y.sort()    # yields [0.9,1.1,1.9,2.1,2.9,3.1,…]

    # ——— compute global vmin/vmax on log(count) ———
    stats = []
    for num in selected_nodes_list:
        stat, _, _, _ = binned_statistic_2d(
            distance_vs_hops_df[distance_vs_hops_df["NumNodes"] == num]["DistanceToGs"],
            distance_vs_hops_df[distance_vs_hops_df["NumNodes"] == num]["HopCount"],
            None, 'count',
            bins=[bins_x, bins_y]
        )
        stats.append(np.log1p(stat))
    vmin = min(s.min() for s in stats)
    vmax = max(s.max() for s in stats)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    # ——————————————————————————————————————

    fig, axs = plt.subplots(1, 2, figsize=(20, 8), sharey=True)

    for i, (selected_num_nodes, rho_label) in enumerate(zip(selected_nodes_list, rho_labels)):
        ax = axs[i]

        # Filter data
        filtered_hops_data = (
            distance_vs_hops_df.groupby(["NumNodes", "HopCount"])
            .filter(lambda group: len(group) >= 0.5 * repititions)
        )
        filtered_data = filtered_hops_data[filtered_hops_data["NumNodes"] == selected_num_nodes]
        x = filtered_data["DistanceToGs"].values
        y = filtered_data["HopCount"].values
        # Heatmap data
        # statistic, x_edge, y_edge, _ = binned_statistic_2d(
        #     filtered_data["DistanceToGs"],
        #     filtered_data["HopCount"],
        #     None, 'count', bins=50
        # )
        # 3) Tell binned_statistic_2d to use exactly those edges:
        statistic, x_edge, y_edge, _ = binned_statistic_2d(
            x, y, None, 'count',
            bins=[bins_x, bins_y]           # 🔧 now uses our custom edges
        )
        statistic_log = np.log1p(statistic)

        # # Plot heatmap
        # ax.imshow(
        #     statistic_log.T, origin='lower',
        #     extent=[x_edge[0], x_edge[-1], y_edge[0], y_edge[-1]],
        #     aspect='auto', cmap=cmap_color
        # )
        # create a mesh of the exact edges
        X, Y = np.meshgrid(x_edge, y_edge)

        pcm = ax.pcolormesh(
            X, Y,
            statistic_log.T,      # note the transpose
            cmap=cmap_color,
            norm=norm,            # ← use shared normalization
            zorder=1,
            shading='flat'        # exact rectangles, no interpolation
        )

        # Ensure that tick labels are drawn on the left of each subplot
        ax.tick_params(axis='y', labelleft=True)

        # ——— ADD HORIZONTAL ARROW FOR A2G RANGE ———
        y_min = y_edge[0]
        total_y = y_edge[-1] - y_edge[0]
        y_pos = y_min + 0.5 * total_y  # 50% above the minimum

        # A2G arrow
        x_a2g_start, x_a2g_end = 0, 370.4
        ax.annotate(
            "", 
            xy=(x_a2g_end, y_pos), 
            xytext=(x_a2g_start, y_pos),
            arrowprops=dict(arrowstyle="<->", color="blue", lw=2)
        )
        ax.text(
            (x_a2g_start + x_a2g_end)/2,
            y_pos - 0.02 * total_y,
            r'$R_{\mathrm{A2G}}$',
            color="blue",
            bbox=dict(facecolor='white', edgecolor='none', pad=0.2),
            ha='center',
            va='top'
        )

        # ——— ADD VERTICAL REFERENCE LINES ———
        base = x_a2g_end               # 370.4 from your arrow
        x_min, x_max = x_edge[0], x_edge[-1]

        # Solid blue lines at base + 100, + 200, …
        n_steps = int((x_max - base) // 100)
        for k in range(0, n_steps + 1):
            x_line = base + 100 * k
            ax.axvline(x_line, color='blue', linestyle='-', linewidth=2)

        # ——— ADD HORIZONTAL ARROW FOR A2A RANGE ———
        # pick two consecutive bars—here base, base+100:
        x_a2a_start = base + 500
        x_a2a_end   = base + 600

        # choose a y‐position a bit below your A2G arrow:
        y_a2a = y_min + 0.075 * total_y  

        ax.annotate(
            "", 
            xy=(x_a2a_end, y_a2a), 
            xytext=(x_a2a_start, y_a2a),
            arrowprops=dict(arrowstyle="<->", color="red", lw=2)
        )
        ax.text(
            (x_a2a_start + x_a2a_end) / 2,
            y_a2a - 0.02 * total_y,
            r'$R_{\mathrm{A2A}}$',
            color="red",
            bbox=dict(facecolor='white', edgecolor='none', pad=0.2),
            ha='center',
            va='top'
        )
        # ————————————————————————————————

        # Dark-green dashed line at 500 km
        if x_min <= 500 <= x_max:
            ax.axvline(500, color='darkgreen', linestyle='--', linewidth=3)
        # ————————————————————————————————
        
        ax.set_xlabel("Distance to Ground Station (km)")
        if i == 0:
            ax.set_ylabel("Hop Count")
        
        # ax.set_ylim(1, 19)  # from 0.9 up to last upper edge
        ax.set_ylim(1, 19)

        # (re)compute centers for ticks if needed:
        y_centers = (bins_y[:-1] + bins_y[1:]) / 2


        ax.xaxis.set_major_locator(ticker.MultipleLocator(250))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(50))
        # y-ticks at odd hop counts only
        max_hop = int(y_edge[-1])
        odd_hops = np.arange(1, max_hop+1, 2)
        ax.yaxis.set_major_locator(ticker.FixedLocator(odd_hops))
        # ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))

        ax.xaxis.grid(True, which='major', linestyle=(0, (5, 10)), linewidth=0.2)
        ax.xaxis.grid(True, which='minor', linestyle=(0, (5, 20)), linewidth=0.1)
        ax.yaxis.grid(True, which='major', linestyle=(0, (5, 10)), linewidth=0.2)
        ax.yaxis.grid(True, which='minor', linestyle=(0, (5, 20)), linewidth=0.1)

        ax.text(0.5, -0.25, rho_label, transform=ax.transAxes,
                ha='center', va='center')

    # … after your loop, where you have `pcm` …
    # 1) shrink the subplots to 95% width
    fig.subplots_adjust(right=0.95)
    # 2) colorbar Axes: start at 96% across, 2% wide, same vertical span
    cbar_ax = fig.add_axes([0.96, 0.11, 0.02, 0.77])
    # 3) draw the shared colorbar there
    cb = fig.colorbar(pcm, cax=cbar_ax, orientation='vertical')
    cb.set_label("log(count)")
    # # ————————————————————————

    # Layout & save
    # plt.tight_layout()
    # plt.subplots_adjust(bottom=0.2)
    plot_pdf_path = os.path.join(plot_distance_dir, 'heatmap_distance_vs_hops_combined.pdf')
    plt.savefig(plot_pdf_path, format='pdf', bbox_inches='tight')
    # plt.show()
    plt.close()


    # Plotting setup
    plt.figure(figsize=(12, 9))
    # capsize = 4
    # markersize = 8
    # lw = 3
    # Group by 'NumNodes' and plot for each 'Radius'
    for radius in grouped_entropy_df['Radius'].unique():
        if radius > 3:
            continue
        # Filter data for the current radius
        radius_data = grouped_entropy_df[grouped_entropy_df['Radius'] == radius]
        
        # Get the color for this radius
        color = radius_colors.get(radius, 'b')  # Default to blue if radius not in dictionary
        
        plt.errorbar(
            radius_data['EquipageFraction'],
            radius_data['Entropy_mean'],
            yerr=radius_data['Entropy_MoE'],
            label=fr'Greedy-{radius}', 
            linestyle='solid',
            marker=radius_markers[radius],
            markersize=markersize,
            markeredgecolor='black',
            lw=lw,
            capsize=capsize,
            color=color
        )

    # Set axis labels and tick formatting
    plt.xlabel('Equipage Fraction')
    plt.ylabel('Normalized Entropy')

    # Configure y-axis locators
    plt.gca().yaxis.set_major_locator(ticker.AutoLocator())
    plt.gca().yaxis.set_minor_locator(ticker.AutoMinorLocator())

    # Configure x-axis locators
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(0.1))

    # Set up grid
    plt.gca().xaxis.grid(True, which='major', linestyle=(0, (5, 10)), linewidth=0.2)
    plt.gca().xaxis.grid(True, which='minor', linestyle=(0, (5, 20)), linewidth=0.1)
    plt.gca().yaxis.grid(True, which='major', linestyle=(0, (5, 10)), linewidth=0.2)
    plt.gca().yaxis.grid(True, which='minor', linestyle=(0, (5, 20)), linewidth=0.1)

    plt.ylim([0, 1])

    # Add a legend
    plt.legend(loc='best')
    plt.tight_layout()

    # Save the plot as a PDF
    plot_pdf_path = os.path.join(plot_entropy_dir, 'Average_entropy.pdf')
    plt.savefig(plot_pdf_path, format='pdf')
    print(f"Plot saved as PDF at: {plot_pdf_path}")

    # Plotting setup
    plt.figure(figsize=(12, 9))
    # capsize = 4
    # markersize = 8
    # lw = 3
    # Group by 'NumNodes' and plot for each 'Radius'
    for radius in grouped_cond_entropy_df['Radius'].unique():
        if radius > 3:
            continue
        # Filter data for the current radius
        radius_data = grouped_cond_entropy_df[grouped_cond_entropy_df['Radius'] == radius]
        
        # Get the color for this radius
        color = radius_colors.get(radius, 'b')  # Default to blue if radius not in dictionary
        
        plt.errorbar(
            radius_data['EquipageFraction'],
            radius_data['Entropy_cond_mean'],
            yerr=radius_data['Entropy_cond_MoE'],
            label=fr'Greedy-{radius}', 
            linestyle='solid',
            marker=radius_markers[radius],
            markersize=markersize,
            markeredgecolor='black',
            lw=lw,
            capsize=capsize,
            color=color
        )

    # Set axis labels and tick formatting
    plt.xlabel('Equipage Fraction')
    plt.ylabel('Normalized Conditional Entropy')

    # Configure y-axis locators
    plt.gca().yaxis.set_major_locator(ticker.AutoLocator())
    plt.gca().yaxis.set_minor_locator(ticker.AutoMinorLocator())

    # Configure x-axis locators
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(0.1))

    # Set up grid
    plt.gca().xaxis.grid(True, which='major', linestyle=(0, (5, 10)), linewidth=0.2)
    plt.gca().xaxis.grid(True, which='minor', linestyle=(0, (5, 20)), linewidth=0.1)
    plt.gca().yaxis.grid(True, which='major', linestyle=(0, (5, 10)), linewidth=0.2)
    plt.gca().yaxis.grid(True, which='minor', linestyle=(0, (5, 20)), linewidth=0.1)

    plt.ylim([0, 1])

    # Add a legend
    plt.legend(loc='best')
    plt.tight_layout()

    # Save the plot as a PDF
    plot_pdf_path = os.path.join(plot_entropy_dir, 'Average_entropy_conditional.pdf')
    plt.savefig(plot_pdf_path, format='pdf')
    print(f"Plot saved as PDF at: {plot_pdf_path}")

    # Set of NumNodes to compare
    selected_nodes_list = [400, 200]
    rho_labels = [r'(a) $\rho = 0.8$', r'(b) $\rho = 0.4$']

    # For each radius, generate a combined figure
    for radius in grouped_average_hops_df['Radius'].unique():
        if radius > 1:
            continue

        # Create a figure with 1 row and 2 columns
        fig, axs = plt.subplots(1, 2, figsize=(20, 8))

        # Plot settings
        # capsize = 4
        # markersize = 8
        # lw = 3
        hop_locator = ticker.MultipleLocator(1)

        for i, (selected_num_nodes, rho_label) in enumerate(zip(selected_nodes_list, rho_labels)):
            ax = axs[i]

            # Filter for current radius and NumNodes
            radius_data = grouped_average_hops_df[
                (grouped_average_hops_df['Radius'] == radius) & 
                (grouped_average_hops_df['NumNodes'] == selected_num_nodes)
            ]

            ax.errorbar(
                radius_data['Hop'],
                radius_data['ProbabilityPositiveAdvance'],
                yerr=radius_data['ProbabilityPositiveAdvanceMoE'],
                label=r'$\pPlusTA$', 
                linestyle='solid',
                marker=radius_markers[1],
                markersize=markersize,
                markeredgecolor='black',
                lw=lw,
                capsize=capsize,
                color=radius_colors[1]
            )

            ax.errorbar(
                radius_data['Hop'],
                radius_data['ProbabilityZeroAdvance'],
                yerr=radius_data['ProbabilityZeroAdvanceMoE'],
                label=r'$\pZeroTA$', 
                linestyle='solid',
                marker=radius_markers[2],
                markersize=markersize,
                markeredgecolor='black',
                lw=lw,
                capsize=capsize,
                color=radius_colors[2]
            )

            ax.errorbar(
                radius_data['Hop'],
                radius_data['ProbabilityNegativeAdvance'],
                yerr=radius_data['ProbabilityNegativeAdvanceMoE'],
                label=r'$\pMinusTA$', 
                linestyle='solid',
                marker=radius_markers[3],
                markersize=markersize,
                markeredgecolor='black',
                lw=lw,
                capsize=capsize,
                color=radius_colors[3]
            )

            ax.errorbar(
                radius_data['Hop'],
                radius_data['ProbabilityFailure'],
                yerr=radius_data['ProbabilityFailureMoE'],
                label=r'$\pFailTA$', 
                linestyle='solid',
                marker=radius_markers[4],
                markersize=markersize,
                markeredgecolor='black',
                lw=lw,
                capsize=capsize,
                # color='black'
                color=radius_colors[4]
            )

            # Axis setup
            ax.set_xlabel('Hops')
            if i == 0:
                ax.set_ylabel('Probability')

            ax.set_ylim([0, 1])
            ax.xaxis.set_major_locator(hop_locator)
            ax.yaxis.set_major_locator(ticker.AutoLocator())
            ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
            ax.xaxis.grid(True, which='major', linestyle=(0, (5, 10)), linewidth=0.2)
            ax.xaxis.grid(True, which='minor', linestyle=(0, (5, 20)), linewidth=0.1)
            ax.yaxis.grid(True, which='major', linestyle=(0, (5, 10)), linewidth=0.2)
            ax.yaxis.grid(True, which='minor', linestyle=(0, (5, 20)), linewidth=0.1)

            ax.legend(loc='best')

            # Add subplot label
            ax.text(0.5, -0.25, rho_label, transform=ax.transAxes,
                    ha='center', va='center')

        # Final layout and save
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.22)
        filename = f'Average_topological_advance_hops_combined_radius={radius}.pdf'
        plot_pdf_path = os.path.join(plot_topo_dir, filename)
        plt.savefig(plot_pdf_path, format='pdf')
        plt.close()
        print(f"Plot saved as PDF at: {plot_pdf_path}")