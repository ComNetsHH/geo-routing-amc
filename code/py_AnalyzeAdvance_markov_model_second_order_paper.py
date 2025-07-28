#!/usr/bin/env python3
"""
py_AnalyzeAdvance_markov_model_second_order_paper.py

Implements a second‑order Markov chain analysis of topological advance 
probabilities in an ad‑hoc airborne network.

Performs:
  - Random network generation and A2A/A2G connectivity setup.
  - Construction of canonical transition matrices for both first‑ and 
    second‑order Markov models.
  - Adjustment of hop‑by‑hop conditional probabilities to enforce 
    absorbing‑state logic.
  - Computation of absorption probabilities and expected times to absorption.
  - Calculation of hop stretch factors based on expected times.
  - Monte Carlo simulation (across node counts and repetitions) via 
    multiprocessing.
Outputs per‑run results for integration with downstream analysis scripts.
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
from sklearn.metrics import median_absolute_error, mean_absolute_error, mean_squared_error
from collections import defaultdict

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
    'text.latex.preamble': r'\usepackage{lmodern}'
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

def update_probabilities_for_hops(
    p_plus_list,
    p_minus_list,
    p_zero_list,
    p_failure_list,
    conditional_probs_list
):
    """
    Enforce failure propagation whenever a branch goes “stuck” at zero probability.

    1. As soon as p_plus == 0 at some hop, all subsequent hops become pure failures.
    2. For each hop, if any conditional branch (TA+, TA0, TA–) has all zero outgoing
       probabilities (and no failure), flip its failure flag on.
    3. Enforce absorbing‑state logic: once a branch is pure failure, subsequent hops
       for that branch must also be pure failures.

    Returns updated lists and conditional_probs_list in the same order.
    """

    # 1) If any p_plus drops to zero, mark that hop and all later hops as failures
    zero_idx = next((i for i, p in enumerate(p_plus_list) if p == 0), None)
    if zero_idx is not None:
        for i in range(zero_idx, len(p_plus_list)):
            p_plus_list[i] = 0
            p_minus_list[i] = 0
            p_zero_list[i] = 0
            p_failure_list[i] = 1

    max_hop = len(p_plus_list)

    # 2) For each hop, enforce absorbing failures in the conditional tables
    for hop in range(1, max_hop + 1):
        cp = conditional_probs_list[hop - 1]

        # Helper to flip a branch to pure failure
        def fail_branch(prefix):
            cp[f'P({prefix}|{prefix})'] = 0
            cp[f'P({prefix}0|{prefix})'] = 0
            cp[f'P({prefix}-|{prefix})'] = 0
            cp[f'P(F|{prefix})'] = 1

        # A) TA+ branch stuck?
        if all(cp[key] == 0 for key in ['P(TA+|TA+)', 'P(TA0|TA+)', 'P(TA-|TA+)', 'P(F|TA+)']):
            fail_branch('TA+')

        # B) TA0 branch stuck?
        if all(cp[key] == 0 for key in ['P(TA+|TA0)', 'P(TA0|TA0)', 'P(TA-|TA0)', 'P(F|TA0)']):
            fail_branch('TA0')

        # C) TA– branch stuck?
        if all(cp[key] == 0 for key in ['P(TA+|TA-)', 'P(TA0|TA-)', 'P(TA-|TA-)', 'P(F|TA-)']):
            fail_branch('TA-')

        # D) If stuck in TA0 (P(TA0|TA0)==1) force failure
        if cp['P(TA0|TA0)'] == 1:
            fail_branch('TA0')

        # E) No transition into TA+ at all → all future hops pure failure
        if (cp['P(TA+|TA+)'] == 0 and
            cp['P(TA+|TA0)'] == 0 and
            cp['P(TA+|TA-)'] == 0):
            for future in range(hop - 1, max_hop):
                conditional_probs_list[future] = {
                    **{k: 0 for k in [
                        'P(TA+|TA+)', 'P(TA0|TA+)', 'P(TA-|TA+)',
                        'P(TA+|TA0)', 'P(TA0|TA0)', 'P(TA-|TA0)',
                        'P(TA+|TA-)', 'P(TA0|TA-)', 'P(TA-|TA-)'
                    ]},
                    **{f'P(F|{b})': 1 for b in ['TA+', 'TA0', 'TA-']}
                }

        # F) If both (P(TA+|TA+) & P(TA+|TA0)) == 0
        #    or (P(TA+|TA+) & P(TA0|TA+)) == 0 → subsequent hops fail
        if ((cp['P(TA+|TA+)'] == 0 and cp['P(TA+|TA0)'] == 0) or
            (cp['P(TA+|TA+)'] == 0 and cp['P(TA0|TA+)'] == 0)):
            for future in range(hop, max_hop):
                conditional_probs_list[future] = {
                    **{k: 0 for k in [
                        'P(TA+|TA+)', 'P(TA0|TA+)', 'P(TA-|TA+)',
                        'P(TA+|TA0)', 'P(TA0|TA0)', 'P(TA-|TA0)',
                        'P(TA+|TA-)', 'P(TA0|TA-)', 'P(TA-|TA-)'
                    ]},
                    **{f'P(F|{b})': 1 for b in ['TA+', 'TA0', 'TA-']}
                }

    return p_plus_list, p_minus_list, p_zero_list, p_failure_list, conditional_probs_list

def build_transition_matrix_by_hops_canonical(
    max_hop,
    p_plus_list,
    p_minus_list,
    p_zero_list,
    p_fail_list,
    conditional_probs_list
):
    """
    build_transition_matrix_by_hops_canonical.py

    Constructs the canonical-form transition matrix for a hop‐by‐hop Markov chain:
      - States: “0” (absorbing success), “1”…“max_hop” (transient), sub‐states i|i+1, i|i, i|i−1, and “F” (absorbing failure).
      - Main states transition by p_plus, p_zero, p_minus, p_fail.
      - Sub‐states transition by the corresponding conditional probabilities.
    Returns:
      Q             Transient→Transient block
      R             Transient→Absorbing block ([“0”, “F”])
      Canonical_T   Full canonical transition matrix
    """

    # —————————————————————————————————————————————
    # 1) Enumerate states in canonical order
    #    0, 1…max_hop, then for each i=2…max_hop: i|i+1, i|i, i|i−1, then F
    # —————————————————————————————————————————————
    states = ["0"] + [str(i) for i in range(1, max_hop + 1)]
    for i in range(2, max_hop + 1):
        states += [f"{i}|{i+1}", f"{i}|{i}", f"{i}|{i-1}"]
    states.append("F")

    # Remove invalid sub‐states at the boundaries
    if max_hop >= 2:
        states = [s for s in states if s != "2|1"]
    if max_hop == 2:
        states = [s for s in states if s != "2|3"]
    if max_hop > 2:
        states = [s for s in states if s != f"{max_hop}|{max_hop+1}"]

    state_index = {s: idx for idx, s in enumerate(states)}
    n = len(states)

    # —————————————————————————————————————————————
    # 2) Initialize full transition matrix
    # —————————————————————————————————————————————
    T = np.zeros((n, n), dtype=float)

    # Absorbing states “0” and “F”
    T[state_index["0"], state_index["0"]] = 1.0
    T[state_index["F"], state_index["F"]] = 1.0

    # “1” → always to “0”
    if max_hop >= 1:
        idx1 = state_index["1"]
        T[idx1, state_index["0"]] = 1.0

    # —————————————————————————————————————————————
    # 3) State “2” and its sub‐states
    # —————————————————————————————————————————————
    if max_hop >= 2:
        idx2 = state_index["2"]
        p_plus_2  = p_plus_list[1]
        p_zero_2  = p_zero_list[1]
        p_minus_2 = p_minus_list[1]
        p_fail_2  = p_fail_list[1]

        # Main state “2”
        if max_hop == 2:
            T[idx2, state_index["1"]]    = p_plus_2
            T[idx2, state_index["2|2"]]  = p_zero_2
            T[idx2, state_index["F"]]    = p_fail_2
        else:
            T[idx2, state_index["1"]]    = p_plus_2
            T[idx2, state_index["2|2"]]  = p_zero_2
            T[idx2, state_index["3|2"]]  = p_minus_2
            T[idx2, state_index["F"]]    = p_fail_2

        # Sub‐state “2|3”
        if max_hop > 2:
            row = state_index["2|3"]
            cp = conditional_probs_list[1]
            T[row, state_index["1"]]    = cp.get("P(TA+|TA+)", 0.0)
            T[row, state_index["2|2"]]  = cp.get("P(TA0|TA+)", 0.0)
            T[row, state_index["3|2"]]  = cp.get("P(TA-|TA+)", 0.0)
            T[row, state_index["F"]]    = cp.get("P(F|TA+)",   0.0)

        # Sub‐state “2|2”
        row = state_index["2|2"]
        cp = conditional_probs_list[1]
        T[row, state_index["1"]]    = cp.get("P(TA+|TA0)", 0.0)
        T[row, state_index["2|2"]]  = cp.get("P(TA0|TA0)", 0.0)
        if max_hop > 2:
            T[row, state_index["3|2"]] = cp.get("P(TA-|TA0)", 0.0)
        T[row, state_index["F"]]    = cp.get("P(F|TA0)",   0.0)

    # —————————————————————————————————————————————
    # 4) States i=3…max_hop−1 and their sub‐states
    # —————————————————————————————————————————————
    for i in range(3, max_hop):
        # Main state “i”
        row_i = state_index[str(i)]
        p_plus_i  = p_plus_list[i-1]
        p_zero_i  = p_zero_list[i-1]
        p_minus_i = p_minus_list[i-1]
        p_fail_i  = p_fail_list[i-1]

        T[row_i, state_index[f"{i-1}|{i}"]]  = p_plus_i
        T[row_i, state_index[f"{i}|{i}"]]    = p_zero_i
        T[row_i, state_index[f"{i+1}|{i}"]]  = p_minus_i
        T[row_i, state_index["F"]]           = p_fail_i

        # Sub‐state i|i+1 (branch from TA+)
        row = state_index[f"{i}|{i+1}"]
        cp = conditional_probs_list[i-1]
        T[row, state_index[f"{i-1}|{i}"]]    = cp.get("P(TA+|TA+)", 0.0)
        T[row, state_index[f"{i}|{i}"]]      = cp.get("P(TA0|TA+)", 0.0)
        T[row, state_index[f"{i+1}|{i}"]]    = cp.get("P(TA-|TA+)", 0.0)
        T[row, state_index["F"]]             = cp.get("P(F|TA+)",   0.0)

        # Sub‐state i|i (branch from TA0)
        row = state_index[f"{i}|{i}"]
        T[row, state_index[f"{i-1}|{i}"]]    = cp.get("P(TA+|TA0)", 0.0)
        T[row, state_index[f"{i}|{i}"]]      = cp.get("P(TA0|TA0)", 0.0)
        T[row, state_index[f"{i+1}|{i}"]]    = cp.get("P(TA-|TA0)", 0.0)
        T[row, state_index["F"]]             = cp.get("P(F|TA0)",   0.0)

        # Sub‐state i|i−1 (branch from TA−)
        row = state_index[f"{i}|{i-1}"]
        T[row, state_index[f"{i-1}|{i}"]]    = cp.get("P(TA+|TA-)", 0.0)
        T[row, state_index[f"{i}|{i}"]]      = cp.get("P(TA0|TA-)", 0.0)
        T[row, state_index[f"{i+1}|{i}"]]    = cp.get("P(TA-|TA-)", 0.0)
        T[row, state_index["F"]]             = cp.get("P(F|TA-)",   0.0)

    # —————————————————————————————————————————————
    # 5) State max_hop and its sub‐states (no i+1 transitions)
    # —————————————————————————————————————————————
    if max_hop > 2:
        i = max_hop
        row_i = state_index[str(i)]
        p_plus_i  = p_plus_list[i-1]
        p_zero_i  = p_zero_list[i-1]
        p_fail_i  = p_fail_list[i-1]

        # Main state “max_hop”
        T[row_i, state_index[f"{i-1}|{i}"]] = p_plus_i
        T[row_i, state_index[f"{i}|{i}"]]   = p_zero_i
        T[row_i, state_index["F"]]          = p_fail_i

        # Sub‐state i|i (branch from TA0)
        row = state_index[f"{i}|{i}"]
        cp = conditional_probs_list[i-1]
        T[row, state_index[f"{i-1}|{i}"]]   = cp.get("P(TA+|TA0)", 0.0)
        T[row, state_index[f"{i}|{i}"]]     = cp.get("P(TA0|TA0)", 0.0)
        T[row, state_index["F"]]            = cp.get("P(F|TA0)",   0.0)

        # Sub‐state i|i−1 (branch from TA−)
        row = state_index[f"{i}|{i-1}"]
        T[row, state_index[f"{i-1}|{i}"]]   = cp.get("P(TA+|TA-)", 0.0)
        T[row, state_index[f"{i}|{i}"]]     = cp.get("P(TA0|TA-)", 0.0)
        T[row, state_index["F"]]            = cp.get("P(F|TA-)",   0.0)

    # —————————————————————————————————————————————
    # 6) Extract Q, R blocks and assemble full canonical matrix
    # —————————————————————————————————————————————
    transient = list(range(1, n-1))
    Q = T[transient][:, transient]
    R = T[transient][:, [0, n-1]]

    Canonical_T = np.zeros_like(T)
    tsize = len(transient)
    Canonical_T[:tsize, :tsize]   = Q
    Canonical_T[:tsize, tsize:]   = R
    Canonical_T[tsize:, tsize:]   = np.eye(2)

    return Q, R, Canonical_T

def calculate_conditioned_matrix(Q, R):
    """
    calculate_conditioned_matrix.py

    Conditions a transient‐state transition matrix on eventual absorption into state “0”.

    Steps:
      1. Compute the fundamental matrix N = (I – Q)^{-1}.
      2. Compute absorption probabilities B = N @ R.
      3. Extract g[k] = Prob(absorb in “0” starting from transient state k).
      4. Reweight Q to obtain Q_tilde[i,j] = Q[i,j] * g[j] / g[i].

    Parameters
    ----------
    Q : np.ndarray, shape (t, t)
        Transient→Transient transition submatrix.
    R : np.ndarray, shape (t, 2)
        Transient→Absorbing transition submatrix ([“0”, “F”]).

    Returns
    -------
    Q_tilde : np.ndarray, shape (t, t)
        Transition matrix among transients, conditioned on absorption in “0”.
    """
    t = Q.shape[0]
    I = np.eye(t)

    # 1) Fundamental matrix (use pseudo-inverse if singular)
    try:
        N = np.linalg.inv(I - Q)
    except np.linalg.LinAlgError:
        N = np.linalg.pinv(I - Q)

    # 2) Absorption probabilities into each absorbing state
    B = N @ R

    # 3) Probability of absorption into “0” from each transient
    g = B[:, 0]

    # 4) Condition Q on absorption in “0”
    Q_tilde = np.zeros_like(Q)
    for i in range(t):
        if g[i] > 0:
            Q_tilde[i, :] = Q[i, :] * g / g[i]

    return Q_tilde

def calculate_expected_times(Q):
    """
    calculate_expected_times.py

    Computes expected number of steps to absorption from each transient state.

    Uses the fundamental matrix N = (I – Q)^{-1} and sums its rows.

    Parameters
    ----------
    Q : np.ndarray, shape (t, t)
        Transient→Transient transition submatrix.

    Returns
    -------
    expected_times : np.ndarray, shape (t,)
        Expected absorption time from each transient state.
    """
    I = np.eye(Q.shape[0])
    N = np.linalg.inv(I - Q)
    expected_times = N.sum(axis=1)
    return expected_times


def calculate_expected_times_conditioned(Q_tilde):
    """
    Calculates the expected times to absorption, conditioned on eventually reaching the desired state.

    Parameters:
    - Q_tilde: Modified transition matrix conditioned on absorption into the desired state.

    Returns:
    - Expected times to absorption for transient states.
    """
    # Fundamental matrix for the conditioned chain
    N_tilde = np.linalg.inv(np.eye(Q_tilde.shape[0]) - Q_tilde)
    # Sum each row to get the expected steps from that state
    expected_times = N_tilde.dot(np.ones(Q_tilde.shape[0]))
    return expected_times

def calculate_absorbing_probabilities(Q, R):
    """
    Compute absorption probabilities from each transient state.

    Parameters
    ----------
    Q : ndarray, shape (m, m)
        Transitions among transient states.
    R : ndarray, shape (m, 2)
        Transitions from transient states to the two absorbing states
        (success “0” and failure “F”).

    Returns
    -------
    B : ndarray, shape (m, 2)
        B[i, j] is the probability that, starting in transient state i,
        the chain is eventually absorbed in absorbing state j.
    """
    I = np.eye(Q.shape[0])
    try:
        N = np.linalg.inv(I - Q)
    except np.linalg.LinAlgError:
        N = np.linalg.pinv(I - Q)
    return N.dot(R)

def reciprocal(values):
    """
    Elementwise reciprocal, mapping non‑positive inputs to NaN.

    Parameters
    ----------
    values : array‑like
        Numeric input values.

    Returns
    -------
    out : ndarray
        out[i] = 1/values[i] if values[i]>0, else NaN.
    """
    arr = np.asarray(values, dtype=float)
    return np.where(arr > 0, 1.0 / arr, np.nan)

def apply_confidence_interval(group):
    """
    Compute summary statistics for a group of simulation runs:
      - Mean and margin‑of‑error (MoE) for Greedy vs. Markov success rates
        and their hop‑stretch factors.
      - Median absolute error between Markov and Greedy rates/stretch.

    Parameters
    ----------
    group : pandas.DataFrame
        Must contain columns:
          - 'GreedyRate', 'SuccessMarkov'
          - 'HopStretchFactor', 'HopStretchMarkov'

    Returns
    -------
    pandas.Series
        {
            'MeanGreedy': mean of group['GreedyRate'],
            'MoEGreedy': MoE of group['GreedyRate'],
            'MeanMarkov': mean of group['SuccessMarkov'],
            'MoEMarkov': MoE of group['SuccessMarkov'],
            'MeanGreedyStretch': mean of group['HopStretchFactor'],
            'MoEGreedyStretch': MoE of group['HopStretchFactor'],
            'MeanMarkovStretch': mean of group['HopStretchMarkov'],
            'MoEMarkovStretch': MoE of group['HopStretchMarkov'],
            'MedianSuccessDiff': median absolute error between Markov & Greedy rates,
            'MoEDifference': same as MedianSuccessDiff,
            'MedianStretchDiff': median absolute error between Markov & Greedy stretch,
            'MoEDifferenceStretch': same as MedianStretchDiff
        }
    """
    # confidence intervals for rates
    mean_greedy, _, moe_greedy = confidence_interval_init(group['GreedyRate'])
    mean_markov, _, moe_markov = confidence_interval_init(group['SuccessMarkov'])

    # confidence intervals for hop‑stretch
    mean_greedy_stretch, _, moe_greedy_stretch = confidence_interval_init(group['HopStretchFactor'])
    mean_markov_stretch, _, moe_markov_stretch = confidence_interval_init(group['HopStretchMarkov'])

    # median absolute errors between Markov and Greedy
    med_success_diff = median_absolute_error(group['SuccessMarkov'], group['GreedyRate'])
    med_stretch_diff = median_absolute_error(group['HopStretchMarkov'], group['HopStretchFactor'])

    return pd.Series({
        'MeanGreedy':               mean_greedy,
        'MoEGreedy':                moe_greedy,
        'MeanMarkov':               mean_markov,
        'MoEMarkov':                moe_markov,
        'MeanGreedyStretch':        mean_greedy_stretch,
        'MoEGreedyStretch':         moe_greedy_stretch,
        'MeanMarkovStretch':        mean_markov_stretch,
        'MoEMarkovStretch':         moe_markov_stretch,
        'MedianSuccessDiff':        med_success_diff,
        'MoEDifference':            med_success_diff,
        'MedianStretchDiff':        med_stretch_diff,
        'MoEDifferenceStretch':     med_stretch_diff
    })

def run_simulation(args):
    """
    Run the Markov-model simulation for a given node count and repetition:
      1. Build the network graph and compute hop counts to the ground station.
      2. Load precomputed advance probabilities (overall and by hops) from CSVs.
      3. Aggregate per-radius lists of p_plus, p_zero, p_minus, p_failure,
         and the corresponding conditional probability dicts.
      4. For each radius:
         a. Adjust probabilities to force failure when necessary.
         b. Build the canonical transition matrix (Q, R, T).
         c. Compute the conditioned transition matrix Q_tilde.
         d. Compute expected times to absorption and absorbing probabilities.
         e. Compute per-node success rates and hop-stretch factors.
      5. Return summary of average Markov success and hop stretch per radius.

    Parameters
    ----------
    args : tuple
        (num_nodes, rep, probability_csv_input, probability_by_hops_csv_input)

    Returns
    -------
    num_nodes : int
    rep : int
    results : dict
        Mapping radius → {'SuccessMarkov': float, 'HopStretchMarkov': float}
    """
    num_nodes, rep, prob_csv, hops_csv = args
    print(f"Running sim for NumNodes={num_nodes}, Rep={rep}")

    # 1) Seed RNG and build network
    np.random.seed(rep)
    area_length, area_side = 1250, 800
    a2a_comm_range, a2g_comm_range = 100, 370.4
    GS_id, GS_pos = 501, (1150, 400)
    G, positions = create_network(
        area_length, area_side, num_nodes,
        a2a_comm_range, a2g_comm_range,
        GS_id, GS_pos
    )
    hop_counts = nx.single_source_shortest_path_length(G, GS_id)
    # Filter only nodes that actually require hops
    hop_counts = {node: h for node, h in hop_counts.items() if h > 0}

    # 2) Load overall and by-hop probability CSVs
    df_overall = pd.read_csv(prob_csv)
    df_hops    = pd.read_csv(hops_csv)

    # 3) Filter for this sim run
    df_overall = df_overall[
        (df_overall['NumNodes'] == num_nodes) &
        (df_overall['Rep'] == rep)
    ]
    df_hops = df_hops[
        (df_hops['NumNodes'] == num_nodes) &
        (df_hops['Rep'] == rep)
    ]

    # 4) Build per-radius lists of probabilities
    by_radius = defaultdict(lambda: {
        'p_plus': [], 'p_zero': [], 'p_minus': [], 'p_failure': [],
        'conditional': []
    })
    # Extract hop-by-hop entries sorted by (Radius, Hop)
    hop_cols = [
        'Radius','Hop',
        'ProbabilityPositiveAdvance','ProbabilityZeroAdvance',
        'ProbabilityNegativeAdvance','ProbabilityFailure',
        'P(TA+|TA+)','P(TA0|TA+)','P(TA-|TA+)','P(F|TA+)',
        'P(TA+|TA0)','P(TA0|TA0)','P(TA-|TA0)','P(F|TA0)',
        'P(TA+|TA-)','P(TA0|TA-)','P(TA-|TA-)','P(F|TA-)'
    ]
    for row in df_hops[hop_cols].sort_values(['Radius','Hop']).to_dict('records'):
        r = row['Radius']
        by_radius[r]['p_plus'].append(row['ProbabilityPositiveAdvance'])
        by_radius[r]['p_zero'].append(row['ProbabilityZeroAdvance'])
        by_radius[r]['p_minus'].append(row['ProbabilityNegativeAdvance'])
        by_radius[r]['p_failure'].append(row['ProbabilityFailure'])
        # store the conditional dictionary for this hop
        cond = {k: row[k] for k in hop_cols if 'P(' in k}
        by_radius[r]['conditional'].append(cond)

    # 5) Collect per-radius overall conditional probabilities
    overall_cond = {
        r: df_overall[df_overall['Radius'] == r].iloc[0].filter(like='P(').to_dict()
        for r in df_overall['Radius'].unique()
    }

    # 6) For each radius, build Markov chain and compute metrics
    results = {}
    for r, prob in by_radius.items():
        p_plus_list    = prob['p_plus']
        p_zero_list    = prob['p_zero']
        p_minus_list   = prob['p_minus']
        p_failure_list = prob['p_failure']
        cond_list      = prob['conditional']

        # ensure lists align
        n = len(p_plus_list)
        if any(len(lst) != n for lst in (p_zero_list, p_minus_list, p_failure_list, cond_list)):
            raise RuntimeError(f"Radius {r}: mismatched list lengths")

        # a) force failure where needed
        p_plus_list, p_minus_list, p_zero_list, p_failure_list, cond_list = \
            update_probabilities_for_hops(
                p_plus_list, p_minus_list, p_zero_list,
                p_failure_list, cond_list
            )

        # b) build canonical transition matrix
        Q, R, _ = build_transition_matrix_by_hops_canonical(
            max_hop=len(p_plus_list),
            p_plus_list=p_plus_list,
            p_minus_list=p_minus_list,
            p_zero_list=p_zero_list,
            p_fail_list=p_failure_list,
            conditional_probs_list=cond_list
        )

        # c) condition on absorption into state 0
        Q_tilde = calculate_conditioned_matrix(Q, R)

        # d) expected times & absorbing probabilities
        exp_times = calculate_expected_times_conditioned(Q_tilde)
        B = calculate_absorbing_probabilities(Q, R)
        absorb_probs = B[:, 0]  # probability of success absorption

        # e) per-node success & stretch
        success_rates = {}
        hop_stretches = {}
        for node, h in hop_counts.items():
            idx = h - 1
            success_rates[node] = absorb_probs[idx]
            hop_stretches[node] = exp_times[idx] / h if exp_times[idx] >= h else 1.0

        # average over nodes
        results[r] = {
            'SuccessMarkov':     np.mean(list(success_rates.values())),
            'HopStretchMarkov':  np.mean(list(hop_stretches.values()))
        }

    return num_nodes, rep, results



if __name__ == "__main__":
    # —————————————————————————————————————————————
    # 1) Settings and parameters
    # —————————————————————————————————————————————
    a2a_comm_range = 100    # A2A Communication range (km)
    # repititions     = 2000 # number of Monte‑Carlo runs used in the paper (HPC can easily handle this)
    repititions     = 5        # for quick local testing; paper results used 2,000 runs
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
    dir_markov     = os.path.join(base_csv_dir, "second_order_model")
    simulation_dir = os.path.join(base_csv_dir, "simulation")

    base_plot_dir = os.path.join(project_root, "results", "plots")
    plot_markov_dir = os.path.join(base_plot_dir, "second_order_model")

    os.makedirs(dir_advance,      exist_ok=True)
    os.makedirs(dir_hops,         exist_ok=True)
    os.makedirs(dir_markov,       exist_ok=True)
    os.makedirs(simulation_dir,   exist_ok=True)
    os.makedirs(plot_markov_dir,  exist_ok=True)

    # —————————————————————————————————————————————
    # 3) CSV filenames & paths
    # —————————————————————————————————————————————
    markov_csv        = f"MarkovModel_SecondOrder_by_hops_{max_num_nodes}.csv"
    grouped_avg_csv   = f"GroupedAverage_MarkovModel_SecondOrder_by_hops_{max_num_nodes}.csv"

    markov_csv_path        = os.path.join(dir_markov,    markov_csv)
    grouped_avg_advance_path = os.path.join(dir_markov, grouped_avg_csv)

    simulation_csv_input       = os.path.join(simulation_dir,  f"simulation_greedy.csv")
    probability_csv_input      = os.path.join(dir_advance,    f"ProbabilityAdvanceConditional_{max_num_nodes}.csv")
    probability_by_hops_input  = os.path.join(dir_hops,       f"ProbabilityAdvance_by_hops_{max_num_nodes}.csv")

    # —————————————————————————————————————————————
    # 4) Load base DataFrame and prepare columns
    # —————————————————————————————————————————————
    df_advance = pd.read_csv(simulation_csv_input)[
        ['NumNodes', 'Rep', 'Radius', 'GreedyRate', 'HopStretchFactor']
    ]

    # ensure Markov columns exist
    for col in ['SuccessMarkov', 'HopStretchMarkov']:
        if col not in df_advance:
            df_advance[col] = np.nan

    # —————————————————————————————————————————————
    # 5) Run simulations in parallel, updating df_advance on the fly
    # —————————————————————————————————————————————
    params = list(product(
        num_node_values,
        range(repititions),
        [probability_csv_input],
        [probability_by_hops_input]
    ))
    ncpus = int(os.environ.get('SLURM_CPUS_PER_TASK', cpu_count()))
    print(f"Using {ncpus} CPUs")


    with Pool(ncpus) as pool:
        for n, rep, avg_dict in pool.imap(run_simulation, params):
            # update SuccessMarkov & HopStretchMarkov per (n,rep,radius)
            for r, props in avg_dict.items():
                m = (df_advance['NumNodes']==n) & (df_advance['Rep']==rep) & (df_advance['Radius']==r)
                df_advance.loc[m, 'SuccessMarkov']     = props['SuccessMarkov']
                df_advance.loc[m, 'HopStretchMarkov']  = props['HopStretchMarkov']
            print(f"Processed sim for NumNodes={n}, Rep={rep}")

    # drop incomplete rows
    df_advance.dropna(inplace=True)

    # reorder for clarity
    df_advance = df_advance[[
        'NumNodes','Rep','Radius',
        'GreedyRate','SuccessMarkov',
        'HopStretchFactor','HopStretchMarkov'
    ]]

    # save the raw Markov results
    df_advance.to_csv(markov_csv_path, index=False)
    print(f"Wrote Markov results to {markov_csv_path}")

    # —————————————————————————————————————————————
    # 6) Group and compute confidence intervals
    # —————————————————————————————————————————————
    df_grouped = (
        pd.read_csv(markov_csv_path)
          .groupby(['NumNodes','Radius'])
          .apply(apply_confidence_interval, include_groups=False)
          .reset_index()
    )
    df_grouped.to_csv(grouped_avg_advance_path, index=False)
    print(f"Saved grouped results with CIs to {grouped_avg_advance_path}")

    # ─────────────────────────────────────────────────────────────────
    # Read grouped averages and compute equipage fraction
    # ─────────────────────────────────────────────────────────────────
    df_grouped = pd.read_csv(grouped_avg_advance_path)
    df_grouped['EquipageFraction'] = df_grouped['NumNodes'] / max_num_nodes


    # -------------------------------------------------------------------
    # Color & marker settings for each radius
    # -------------------------------------------------------------------
    radius_colors = {
        1: 'k',       # Black
        2: '#298C8C', # Teal
        3: 'olive',   # Olive
        4: '#A00000', # Red
    }
    radius_markers = {
        1: 'o',
        2: 's',
        3: '^',
        4: 'v',
    }

    # Create a figure with 1 row and 2 columns
    fig, axs = plt.subplots(1, 2, figsize=(20, 8))

    # Plot settings
    capsize = 4
    markersize = 9
    lw = 4

    # ======== Plot 1: Success Ratio (Second-Order Model) ========
    ax = axs[0]
    for radius in df_grouped['Radius'].unique():
        if radius > 3:
            continue
        radius_data = df_grouped[df_grouped['Radius'] == radius]
        color = radius_colors.get(radius, 'b')

        ax.errorbar(
            radius_data['EquipageFraction'],
            radius_data['MeanGreedy'],
            yerr=radius_data['MoEGreedy'],
            label=f'Greedy-{radius} (sim.)',
            linestyle='solid',
            marker=radius_markers[radius],
            markersize=markersize,
            markeredgecolor='black',
            lw=lw,
            capsize=capsize,
            color=color
        )

        ax.errorbar(
            radius_data['EquipageFraction'],
            radius_data['MeanMarkov'],
            yerr=radius_data['MoEMarkov'],
            label=f'Greedy-{radius} (model)',
            linestyle='--',
            marker=radius_markers[radius],
            markersize=markersize,
            markeredgecolor='black',
            lw=lw,
            capsize=capsize,
            color=color
        )

    ax.set_xlabel('Equipage Fraction')
    ax.set_ylabel('Success Ratio')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.yaxis.set_major_locator(ticker.AutoLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.xaxis.grid(True, which='major', linestyle=(0, (5, 10)), linewidth=0.2)
    ax.xaxis.grid(True, which='minor', linestyle=(0, (5, 20)), linewidth=0.1)
    ax.yaxis.grid(True, which='major', linestyle=(0, (5, 10)), linewidth=0.2)
    ax.yaxis.grid(True, which='minor', linestyle=(0, (5, 20)), linewidth=0.1)
    ax.legend(loc='best')

    # Add annotation below first subplot
    ax.text(0.5, -0.25, '(a) Success Ratio', transform=ax.transAxes,
            ha='center', va='center')

    # ======== Plot 2: Median Absolute Error (Second-Order Model) ========
    ax = axs[1]
    for radius in df_grouped['Radius'].unique():
        if radius > 3:
            continue
        radius_data = df_grouped[df_grouped['Radius'] == radius]
        color = radius_colors.get(radius, 'b')

        ax.errorbar(
            radius_data['EquipageFraction'],
            radius_data['MedianSuccessDiff'],
            label=f'Greedy-{radius}',
            linestyle='solid',
            marker=radius_markers[radius],
            markersize=markersize,
            markeredgecolor='black',
            lw=lw,
            capsize=capsize,
            color=color
        )

    ax.set_xlabel('Equipage Fraction')
    ax.set_ylabel('Median Absolute Error')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.yaxis.set_major_locator(ticker.AutoLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.xaxis.grid(True, which='major', linestyle=(0, (5, 10)), linewidth=0.2)
    ax.xaxis.grid(True, which='minor', linestyle=(0, (5, 20)), linewidth=0.1)
    ax.yaxis.grid(True, which='major', linestyle=(0, (5, 10)), linewidth=0.2)
    ax.yaxis.grid(True, which='minor', linestyle=(0, (5, 20)), linewidth=0.1)
    ax.legend(loc='best')

    # Add annotation below second subplot
    ax.text(0.5, -0.25, '(b) Median Absolute Error', transform=ax.transAxes,
            ha='center', va='center')

    # Final layout and save
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.22)  # For subplot labels
    combined_pdf_path = os.path.join(plot_markov_dir, 'Greedy_vs_Markov_SucessRate_Combined_second_order.pdf')
    plt.savefig(combined_pdf_path, format='pdf')
    print(f"Combined annotated plot saved as PDF at: {combined_pdf_path}")
    plt.close()

    # Create a figure with 1 row and 2 columns
    fig, axs = plt.subplots(1, 2, figsize=(20, 8))

    # Plot settings
    capsize = 4
    markersize = 9
    lw = 4

    # ======== Plot 1: Hop Stretch Factor (Second-Order Model) ========
    ax = axs[0]
    for radius in df_grouped['Radius'].unique():
        if radius > 3:
            continue
        radius_data = df_grouped[df_grouped['Radius'] == radius]
        color = radius_colors.get(radius, 'b')

        ax.errorbar(
            radius_data['EquipageFraction'],
            radius_data['MeanGreedyStretch'],
            yerr=radius_data['MoEGreedyStretch'],
            label=f'Greedy-{radius} (sim.)',
            linestyle='solid',
            marker=radius_markers[radius],
            markersize=markersize,
            markeredgecolor='black',
            lw=lw,
            capsize=capsize,
            color=color
        )

        ax.errorbar(
            radius_data['EquipageFraction'],
            radius_data['MeanMarkovStretch'],
            yerr=radius_data['MoEMarkovStretch'],
            label=f'Greedy-{radius} (model)',
            linestyle='--',
            marker=radius_markers[radius],
            markersize=markersize,
            markeredgecolor='black',
            lw=lw,
            capsize=capsize,
            color=color
        )

    ax.set_xlabel('Equipage Fraction')
    ax.set_ylabel('Hop Stretch Factor')
    ax.set_ylim([1, 1.08])
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.yaxis.set_major_locator(ticker.AutoLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.xaxis.grid(True, which='major', linestyle=(0, (5, 10)), linewidth=0.2)
    ax.xaxis.grid(True, which='minor', linestyle=(0, (5, 20)), linewidth=0.1)
    ax.yaxis.grid(True, which='major', linestyle=(0, (5, 10)), linewidth=0.2)
    ax.yaxis.grid(True, which='minor', linestyle=(0, (5, 20)), linewidth=0.1)
    ax.legend(loc='best')
    ax.text(0.5, -0.25, '(a) Hop Stretch Factor', transform=ax.transAxes,
            ha='center', va='center')

    # ======== Plot 2: Median Absolute Error (Hop Stretch) ========
    ax = axs[1]
    for radius in df_grouped['Radius'].unique():
        if radius > 3:
            continue
        radius_data = df_grouped[df_grouped['Radius'] == radius]
        color = radius_colors.get(radius, 'b')

        ax.errorbar(
            radius_data['EquipageFraction'],
            radius_data['MedianStretchDiff'],
            label=f'Greedy-{radius}',
            linestyle='solid',
            marker=radius_markers[radius],
            markersize=markersize,
            markeredgecolor='black',
            lw=lw,
            capsize=capsize,
            color=color
        )

    ax.set_xlabel('Equipage Fraction')
    ax.set_ylabel('Median Absolute Error')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.yaxis.set_major_locator(ticker.AutoLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.xaxis.grid(True, which='major', linestyle=(0, (5, 10)), linewidth=0.2)
    ax.xaxis.grid(True, which='minor', linestyle=(0, (5, 20)), linewidth=0.1)
    ax.yaxis.grid(True, which='major', linestyle=(0, (5, 10)), linewidth=0.2)
    ax.yaxis.grid(True, which='minor', linestyle=(0, (5, 20)), linewidth=0.1)
    ax.legend(loc='best')
    ax.text(0.5, -0.25, '(b) Median Absolute Error', transform=ax.transAxes,
            ha='center', va='center')

    # Final layout and save
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.22)
    combined_pdf_path = os.path.join(plot_markov_dir, 'Greedy_vs_Markov_HopStretch_Combined_second_order.pdf')
    plt.savefig(combined_pdf_path, format='pdf')
    print(f"Combined annotated plot saved as PDF at: {combined_pdf_path}")
    plt.close()
