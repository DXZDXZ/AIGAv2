import numpy as np


def gen_coords(n_nodes):
    coords = []
    num = 0
    while num < n_nodes:
        tag = False
        x_coord, y_coord = np.random.rand(), np.random.rand()
        for coord in coords:
            if x_coord == coord[0] and y_coord == coord[1]:
                tag = True
        if not tag:
            coords.append([x_coord, y_coord])
            num += 1
        else:
            continue
    coords = np.array(coords)
    return coords


def calc_disMat(coords):
    n_nodes = len(coords)
    disMat = np.zeros((n_nodes, n_nodes), dtype=np.float)

    for i in range(n_nodes):
        disMat[i] = np.sum((coords[i] - coords)**2, axis=1)**0.5
    return disMat


def calc_length(seq, disMat):
    length = 0.0
    for i in range(len(seq) - 1):
        length += disMat[seq[i]][seq[i+1]]
    length += disMat[seq[-1], seq[0]]
    return length
