from functions import *

# Import MAHLERAN's sediment connectivity on grassland and shrubland
FC_mahleran_grass = np.loadtxt('landscape_data/p1_rainA_highsm_dschg.asc', skiprows=6)[1:-1,1:-1]
FC_mahleran_shrub = np.loadtxt('landscape_data/p4_rainA_highsm_dschg.asc', skiprows=6)[1:-1,1:-1]

# Import vegetation plots of grassland and shrubland
vegetation_grass = np.loadtxt('landscape_data/p1vegcover.asc', skiprows=6)[1:-1,1:-1]
vegetation_shrub = np.loadtxt('landscape_data/p4vegcover.asc', skiprows=6)[1:-1,1:-1]

# Import topography of grassland and shrubland
topography_grass = np.loadtxt('landscape_data/p1dem.asc', skiprows=6)[1:-1,1:-1]
topography_shrub = np.loadtxt('landscape_data/p4dem.asc', skiprows=6)[1:-1,1:-1]

# Initialise lattice
x1, x2 = 20, 60
G, not_sinks = tgrid(x1,x2)
G_prob = generate_probabilistic_lattice(G, topography_shrub)
propagation_probability_shrub = compute_propagation_probability(G_prob, topography_shrub)
propagation_probability_shrub_edge = compute_propagation_probability_edge(G_prob, topography_shrub)
#steepest_successor_shrub = identify_deterministic_successor(G_prob, topography_shrub)
#G_deter = generate_deterministic_lattice(G_prob, steepest_successor_shrub)

# Compute structural connectivity (type: dict) and initialise functional connectivity (type: dict)
SC = compute_SC(G_prob, propagation_probability_shrub_edge)
FC = {i:0 for i in G}

# Insert simulation parameters (number of runs and number of steps per run)
nb_runs, nb_steps = 1, 10000

# Sandpile simulation according to probabilistic descent, nb_runs simulation of nb_steps time steps according to deterministic descent
S, coupling_shrub, current, branches, active, sizes = initialisation(G_deter)
for run in range(nb_runs):
    S, coupling_shrub_run, current, branches, active, sizes = reinitialisation(coupling_shrub)
    for t in range(nb_steps):
        S, coupling_shrub_run, current, branches, active, sizes = sandpile(G_prob, vegetation_shrub, propagation_probability_shrub, S, coupling_shrub_run, current, branches, active, sizes, not_sinks)
    # compute functional connectivity after nb_steps time steps
    FC_run = compute_FC(coupling_shrub_run)
    # increment functional connectivity from the previous run
    for i in G:
        FC[i] += FC_run[i]

# compute final SC-FC correlation
SCFC_correlation = SCFC(SC, FC, not_sinks)
