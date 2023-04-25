import numpy as np

from distance import select_closest

def generate_network(size):
    """
    Generate a neuron network of a given size.

    Return a vector of two dimensional points in the interval [0,1].
    """
    return np.random.rand(size, 2)

def get_neighborhood(center, radix, domain):
    """Get the range gaussian of given radix around a center index."""

    # Impose an upper bound on the radix to prevent NaN and blocks
    if radix < 1:
        radix = 1

    # Compute the circular network distance to the center
    deltas = np.absolute(center - np.arange(domain))
    distances = np.minimum(deltas, domain - deltas)

    # Compute Gaussian distribution around the given center
    return np.exp(-(distances*distances) / (2*(radix*radix)))

def get_route(nodes, network):
    """Return the route computed by a network."""
    nodes['winner'] = nodes[['x', 'y']].apply(
        lambda c: select_closest(network, c),
        axis=1, raw=True)

    return nodes.sort_values('winner').index

def get_drone_route(nodes, network, start_idx = 0): 
    """Return the route computed by a network.""" 
    nodes['winner'] = nodes[['x', 'y']].apply( 
        lambda c: select_closest(network, c), 
        axis=1, raw=True) 

    all_idc = nodes.sort_values('winner').index

    # find the index of the start_idx in the sorted list
    start_idx = np.where(all_idc == start_idx)[0][0]

    # rotate the list so that the start_idx is at the beginning
    rolled_idx = np.roll(all_idc, -start_idx)

    # remove the last element, which is the same as the first
    return rolled_idx[:-1]
