#### Modelling soil erosion connectivity using a sandpile framework

/landscape_data contains grassland and shrubland empirical data: landscape topography, vegetation and sediment connectivity from MAHLERAN

functions.py contains all the functions used in the manuscript, which allows the reader to:
- import the empirical data from /landscape_data
- generate a baseline lattice
- adapt the baseline lattice to the imported topography, assuming either a probabilistic or a deterministic descent
- distribute vegetation on a lattice according to a given vegetation cover and clustering
- generate topography from a vegetation distribution
- compute structural connectivity from the topography
- run a sandpile simulation, and compute the resulting functional connectivity

main.py guides the reader from lattice generation to a single simulation of the sandpile model assuming probabilistic descent on shrubland (field data)

