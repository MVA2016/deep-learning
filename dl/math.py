import numpy as np


def eigh(matrix):
    '''
    Compute eigenvectors and eigenvalues (val_i, vec_i)
    of a Hermitian matrix (ie A^* = A).
    They verify A vec_i = val_i vec_i

    Parameters
    ----------
    matrix: ndarray
        hermitian matrix

    Returns
    -------
    tuple(ndarray, ndarray)
        Returns eigenvalues, largest first and corresponding eigenvectors (each column is an eigenvector)
    '''
    eigval, eigvec = np.linalg.eigh(matrix)
    tmp_eigval, tmp_eigvec_T = zip(*sorted(zip(eigval, eigvec.T), key=lambda (a,b): a, reverse=True))
    eigval, eigvec = np.asarray(tmp_eigval), np.asarray(tmp_eigvec_T).T
    return eigval, eigvec


# vertical & horizontal size of lattice
#sv, sh = 4, 4


def get_connection_matrix(sv=4, sh=4, J=0.5):
    """ Return the matrix with all connections (Js)
    
    Parameters
    ----------
    sv: int
        vertical size
    sh : int
        horizontal size
    
    Returns
    -------
    ndarray 
        Returns the grid-based neighbors matrix(sv*sh,sv*sh)
    """
    # matrix containing all possible connections
    Js = np.zeros((sv * sh, sv * sh))
    # connect only grid-based neighbors
    for h in range(sh):
        for v in range(sv):
            neighs = []
            pt = h * sv + v
            #
            if v > 0:
                neighs += [pt - 1]
            if v < sv - 1:
                neighs += [pt + 1]
            if h > 0:
                neighs += [(h - 1) * sv + v]
            if h < sh - 1:
                neighs += [(h + 1) * sv + v]
            Js[pt, neighs] = J
            Js[neighs, pt] = J
    return Js

Js = get_connection_matrix()
