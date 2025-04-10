import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from operator import mul
from functools import reduce
from scipy.stats import pearsonr
import pickle, random, itertools, sys
from math import log
from collections import Counter

def generate_baseline_lattice(x, y):
    '''
    Output baseline lattice (independent of topography) and the outflow layer of nodes
    '''
    g = nx.grid_2d_graph(x,int(y))
    g.remove_edges_from([((i1,0),(i1+1,0)) for i1 in range(x-1)])
    g = g.to_directed()
    g.remove_edges_from([((i1,i2),(i1,i2+1)) for i1,i2 in itertools.product(range(x), range(int(y)-1))])
    g.add_edges_from([((0,i2),(x-1,i2)) for i2 in range(1,int(y))]+[((x-1,i2),(0,i2)) for i2 in range(1,int(y))])
    s_ = [(i, 0) for i in range(x)]
    return g, [i for i in g if i not in s_]

def initialisation(g):
    '''
    Create all the variables required for the sandpile simulation
    state: dictionnary of the number of particles (value) on each node (key)
    coupling: dictionary of the distances covered by particles travelled from a node (first index) to another (second index)
    current_av: dictionary of the nodes (key) involved in the current avalanche, and at which point they received a particle (value)
    branches: dictionary of the source (key) and target (value) nodes of the currently occuring avalanche
    new_active: list of the nodes currently toppling, this is an empty list during the accumulation phase
    size: list of avalanche sizes
    ordlist: ordered list of the lattice node to be used to generate ordlist
    '''
    state, coupling, current_av, branches, new_active, size, ordlist = {}, {}, {}, {}, [], [], sorted(list(g), key=lambda a:a[1], reverse=True)
    for i in g:
        state[i], coupling[i] = 0, {}
    for i,j in itertools.combinations(ordlist, 2):
        if nx.has_path(g,i,j):
            coupling[i][j] = []
        if i[1] == j[1]:
            if nx.has_path(g,j,i):
                coupling[j][i] = []
    return state, coupling, current_av, branches, new_active, size

def sandpile_probabilistic_descent(g, v, ep, state, coupling, current_av, branches, old_active, size, not_sinks):
    '''
    Simulate one time step of the sandpile model following probabilistic descent
    g: lattice networkx variable
    v: dictionary of vegetation density (value) per node (key)
    ep: dictionary of propagation proabilities from a node (key) to its neighbors (value)
    old active: list of previously active nodes = receive particles either during either the accumulation or relaxation phase
    '''
    new_active, unstable = [], [node for node in old_active if state[node] >= g.out_degree(node)]
    if len(unstable):
        for node1 in unstable:
            nb_partcl, state[node1] = state[node1], 0
            if len(current_av) == 0:
                current_av[node1] = 0
            if g.out_degree(node1):
                spreading_scheme = Counter(random.choices(list(g.successors(node1)), weights=ep[(node1)], k=nb_partcl))
                for node2 in spreading_scheme:
                    state[node2] += np.sum(np.random.rand(spreading_scheme[node2]) > v[node2]/2)
                    current_av[node2] = current_av[node1]+1
                    if node2 not in branches:
                        branches[node2] = [node1]
                    elif node1 not in branches[node2]:
                        branches[node2].append(node1)
                    # record all the previous sources of the new incident node (at least, from a new path)
                    origin, all_sources, sources, counted = list(current_av.keys())[0], [node1], [node1], []
                    while origin not in sources:
                        new_sources = []
                        for source in sources:
                            new_sources += branches[source]
                        new_sources = list(set(new_sources))
                        all_sources += new_sources
                        sources = new_sources
                    # record couplings between new incident node and all its previous sources
                    for source in all_sources: # collapse 'all_sources' to avoid redundance of nodes
                        coupling[source][node2].append(current_av[node2]-current_av[source])
                    if node2 not in new_active:
                        new_active.append(node2)
    else:
        if len(current_av):
            size.append(len(current_av)-1)
        current_av, branches = {}, {}
        node2 = random.choice(not_sinks)
        if random.random() > v[node2]/2:
            state[node2] += 1
            new_active.append(node2)
    return state, coupling, current_av, branches, new_active, size

def sandpile_deterministic_descent(g, v, state, coupling, current_av, branches, old_active, size, exit_, not_sinks):
    '''
    Simulate one time step of the sandpile model following deterministic descent
    g: lattice networkx variable, following only steepest descent directions
    '''
    unstable, new_active = [], []
    for node in old_active:
        if state[node] > g.out_degree(node):
            unstable.append(node)
    if len(unstable):
        for node1 in unstable:
            nb_partcl, state[node1] = state[node1], 0
            if len(current_av) == 0:
                current_av[node1] = 0
            if g.out_degree(node1):
                node2 = list(g.successors(node1))[0]
                state[node2] += np.sum(np.random.rand(nb_partcl) > v[node2]/2)
                current_av[node2] = current_av[node1]+1
                if node2 not in branches:
                    branches[node2] = [node1]
                elif node1 not in branches[node2]:
                    branches[node2].append(node1)
                # record all the previous sources of the new incident node (at least, from a new path)
                origin, all_sources, sources, counted = list(current_av.keys())[0], [node1], [node1], []
                while origin not in sources:
                    new_sources = []
                    for source in sources:
                        new_sources += branches[source]
                    new_sources = list(set(new_sources))
                    all_sources += new_sources
                    sources = new_sources
                # record couplings between new incident node and all its previous sources
                for source in all_sources: # collapse 'all_sources' to avoid redundance of nodes
                    coupling[source][node2].append(current_av[node2]-current_av[source])
                if node2 not in new_active:
                    new_active.append(node2)
            else:
                exit_ += nb_partcl
    else:
        if len(current_av):
            size.append(len(current_av))
        current_av, branches = {}, {}
        node2 = random.choice(not_sinks)
        if random.random() > v[node2]/2:
            state[node2] += 1
            new_active.append(node2)
    return state, coupling, current_av, branches, new_active, size, exit_

def sandpile_deterministic_descent_sediment_yield_experiment(g, v, state, current_av, branches, old_active, nb_rain_, exit_, p_rain_, not_sinks):
    '''
    Simulate one time step of the sandpile model following deterministic descent, specific to the sediment yield numerical experiment, see Section 3.1.2
    Here, we use neither 'coupling' nor 'size', as we interested in neither SC-FC correlations nor avalanche size distributions, but only the system response in terms of the number of particles exiting the lattice through its outflow layer (sandpile sediment yield)
    exit_: total number of exiting particles
    nb_rain_: counts the number of accumulated particles
    p_rain_: probability for a particle to be added during the accumulation phase, simulates rainfall intensity
    '''
    new_active, unstable = [], [node for node in old_active if state[node] >= g.out_degree(node)]
    if len(unstable):
        for node1 in unstable:
            nb_partcl, state[node1] = state[node1], 0
            if len(current_av) == 0:
                current_av[node1] = 0
            if g.out_degree(node1):
                node2 = list(g.successors(node1))[0]
                state[node2] += np.sum(np.random.rand(nb_partcl) > v[node2]/2)
                current_av[node2] = current_av[node1]+1
                if node2 not in branches:
                    branches[node2] = [node1]
                elif node1 not in branches[node2]:
                    branches[node2].append(node1)
                if node2 not in new_active:
                    new_active.append(node2)
            else:
                exit_ += nb_partcl
    elif random.random() < p_rain_:
        current_av, branches, nb_rain_ = {}, {}, nb_rain_+1
        node2 = random.choice(not_sinks)
        if random.random() > v[node2]/2:
            state[node2] += 1
            new_active.append(node2)
    return state, current_av, branches, new_active, nb_rain_, exit_

def compute_SC(g,epmap):
    '''
    Compute structural connectivity
    g: lattice networkx variable, following probabilistic or deterministic descent directions
    epmap: dictionary of propagation probabilities (value) of each edge (key)
    '''
    gconn = deepcopy(g)
    gconn.remove_edges_from(list(g.edges()))
    for j in range(60):
        next_layer_successors, same_layer_successors = {v1:[v2 for v2 in g.successors(v1) if v2[1]==v1[1]-1] for v1 in [(i,j) for i in range(20)]}, {v1:[v2 for v2 in g.successors(v1) if v2[1]==v1[1]] for v1 in [(i,j) for i in range(20)]}
        for i in range(20):
            v1 = (i,j)
            gconn.add_weighted_edges_from([(v1,v2,epmap[(v1,v2)]) for v2 in next_layer_successors[v1]])
            gconn.add_weighted_edges_from(sum([[(v1,v3,gconn[v2][v3]['weight']) for v3 in gconn.successors(v2)] for v2 in next_layer_successors[v1]],[]))
        for v1 in sort_layer(same_layer_successors):
            gconn.add_weighted_edges_from([(v1,v2,epmap[(v1,v2)]) for v2 in same_layer_successors[v1]])
            gconn.add_weighted_edges_from(sum([[(v1,v3,gconn[v2][v3]['weight']) for v3 in gconn.successors(v2)] for v2 in same_layer_successors[v1]],[]))
    return {node:gconn.in_degree(node, weight='weight') for node in g}

def compute_FC(coupling):
    '''
    Compute functional connectivity using 'coupling' as input
    '''
    fc, fci = {}, {}
    for i in coupling:
        fci[i] = []
    for i in coupling:
        for j in coupling[i]:
            fci[j] += coupling[i][j]
    for i in coupling:
        fc[i] = 0
        if len(fci[i]):
            fc[i] += np.var(fci[i])
    return fc

def nx_to_mat(dict_lat):
    global x, y
    '''
    Convert lattice dictionnary to matrix for visualisation
    x: input dictionary
    '''
    mat = np.zeros((y,x), float)
    for i in range(y):
        for j in range(x):
            mat[i,j] = dict_lat[(j,y-1-i)]
    return mat

def reinitialisation(coupling):
    '''
    Create all the variables required for the sandpile simulation
    Unlike initialisation(), this function skips 'coupling' and allows faster several successive simulations
    '''
    state, coupling_empty, current_av, branches, new_active, size = {}, {}, {}, {}, [], []
    for i in coupling:
        state[i], coupling_empty[i] = 0, {}
        for j in coupling[i]:
            coupling_empty[i][j] = []
    return state, coupling_empty, current_av, branches, new_active, size

def generate_probabilistic_lattice(g, e):
    '''
    Generate lattice (type: nx.diGraph) according to probabilistic descent
    g: baseline lattice
    e: topography
    '''
    geff = nx.DiGraph()
    geff.add_nodes_from(g.nodes)
    for i in g:
        for j in g.successors(i):
            if e[i] >= e[j]:
                geff.add_edge(i,j)
    return geff

def identify_deterministic_successor(g, e):
    '''
    Generate dictionnary identifying the successor (value) of each node (key) according to deterministic descent
    g: lattice according to probabilistic descent
    e: topography dictionnary
    '''
    est = {}
    for i in g:
        next_nodes = list(g.successors(i))
        if len(next_nodes):
            elev_nodes = np.array([e[k] for k in next_nodes])
            min_elev = np.where(elev_nodes==np.min(elev_nodes))[0]
            if len(min_elev)>1:
                est[i] = (i[0],i[1]-1)
            else:
                est[i] = next_nodes[min_elev[0]]
    return est

def generate_deterministic_lattice(g, est):
    '''
    Generate lattice (type: nx.diGraph) according to deterministic descent
    g: lattice according to probabilistic descent
    est: dictionary of deterministic successors, output of the function identify_deterministic_successor()
    '''
    gst = deepcopy(g)
    gst.remove_edges_from(list(g.edges))
    for i in est:
        _ = gst.add_edge(i, est[i])
    return gst

def compute_propagation_probability(g, e):
    '''
    Create dictionnary computing the propagation probabilities (value) of each node (key)
    '''
    ep = {}
    for i in g:
        ep[i] = [max(0,e[i]-e[j]) for j in g.successors(i)]
        if ep[i]==[0] and len(list(g.successors(i)))==1:
            ep[i] = [1]
    return ep

def compute_propagation_probability_edge(g, e):
    '''
    Create dictionnary computing the propagation probabilities (value) of each edge (key)
    '''
    epmap = {}
    for i in g:
        epi = {j: max(0,e[i]-e[j]) for j in g.successors(i)}
        sum_probs = sum(epi.values())
        for j in epi:
            if epi[j]:
                epmap[(i,j)] = epi[j]/sum_probs
        if epi==[0] and len(list(g.successors(i)))==1:
            epmap[(i,list(g.successors(i))[0])] = 1
    return epmap

def compute_SCFC(sc, fc, not_sinks):
    '''
    Compute the SC-FC correlation
    sc: dictionary of structural connectivity, output of the function compute_SC()
    fc: dictionary of structural connectivity, output of the function compute_FC()
    not_sinks: list of nodes of the outflow later, output of the function generate_baseline_lattice()
    '''
    return pearsonr(list(sc[i] for i in not_sinks), list(fc[i] for i in not_sinks))[0]

def distribute_vegetation(g, cover, pclust):
    '''
    Generate a vegetation plot v, dictionary of vegetation density (value) per node (key)
    cover: ratio of vegetated nodes, real number between 0 and 1
    pclust: clustering probability of the vegetated nodes, real number between 0 and 1
    '''
    v = {}
    for i in g:
        v[i] = 0
    free_nodes, nb_veg_final, nb_veg = set(list(g)), int(len(g)*cover), 0
    node = random.choice(list(free_nodes))
    while nb_veg < nb_veg_final:
        v[node] = 1#random.random()
        free_nodes.remove(node)
        nb_veg += 1
        if random.random() < pclust:
            i,j = node
            free_neighbors = {(i,j+1), (i,j-1), (i-1,j), (i+1,j), (i-1,j+1), (i+1,j+1), (i+1,j-1), (i-1,j+1)} & free_nodes
            if free_neighbors:
                node = random.choice(list(free_neighbors))
            else:
                node = random.choice(list(free_nodes))
        else:
            node = random.choice(list(free_nodes))
    return v

def generate_topography(g, v, ep, a, a_std, b):
    '''
    Generate topography from a generated vegetation plot
    g: baseline lattice
    v: generated vegetation plot, output of distribute_vegetation()
    ep: propagation probability, output of compute_propagation_probability()
    a: slope of the linear regression between vegetation score and microtopography
    a_std: standard deviation of a
    b: intercept of the linear regression between vegetation score and microtopography
    '''
    v_score, mt, e, g_x, g_y = {}, {}, {}, list(g)[-1][0], list(g)[-1][1]
    for i in range(1,19):
        for j in range(1,59):
            v_score[i,j] = v[i,j] + v[i-1,j]+ v[i+1,j] + v[i,j-1] + v[i,j+1]
            v_score[0,j] = 2*v[0,j] + v[1,j] + v[0,j-1] + v[0,j+1]
            v_score[19,j] = 2*v[19,j] + v[18,j] + v[19,j-1] + v[19,j+1]
        v_score[i,0] = 2*v[i,0] + v[i-1,0] + v[i+1,0] + v[i,1]
        v_score[i,59] = 2*v[i,59] + v[i-1,59] + v[i+1,59] + v[i,58]
    v_score[0,59] = 3*v[0,59] + v[1,59] + v[0,58]
    v_score[19,59] = 3*v[19,59] + v[18,59] + v[19,58]
    v_score[0,0] = 3*v[0,0] + v[1,0] + v[0,1]
    v_score[19,0] = 3*v[19,0] + v[18,0] + v[19,1]
    for i,j in g:
        mt[i,j] = v_score[i,j]*a + (2*random.random()-1)*a_std + b
        e[i,j] = mt[i,j] + np.mean([ep[k,j] for k in range(20)])
    # outputs e (topography) and mt (microtopography = topography minus planar elevation)
    return e, mt
