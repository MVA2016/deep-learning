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
