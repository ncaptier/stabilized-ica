import warnings
from typing import Optional, Tuple, Union

import numpy as np
from scipy.sparse import issparse, spmatrix
from scipy.sparse.linalg import LinearOperator, svds
from sklearn.decomposition import PCA, IncrementalPCA, TruncatedSVD
from sklearn.utils import check_array, check_random_state
from sklearn.utils.extmath import svd_flip

"""The following code is inspired by the scanpy.tl.pca module (
https://scanpy.readthedocs.io/en/stable/api/scanpy.tl.pca.html). 

Copyright (c) 2017 F. Alexander Wolf, P. Angerer, Theis Lab
All rights reserved."""


def whitening(
        X: Union[np.ndarray, spmatrix],
        n_components: int,
        svd_solver: str,
        chunked: bool,
        chunk_size: Union[int, None],
        zero_center: bool,
        random_state: Optional[Union[int, np.random.RandomState]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """ Whiten data (i.e. transform variables into a set of new uncorrelated and unit-variance variables) and reduce
    dimension trhough a PCA-like approach. This function handles array-like formats as well as sparse matrices.
    
    Parameters
    ----------
    X : 2D ndarray or spmatrix, shape (n_samples , n_features)
        
    n_components : int
        number of pricipal components to compute. If None, n_components = min(X.shape)
        
    svd_solver : str {‘auto’, ‘full’, ‘arpack’, ‘randomized’ , 'lobpcg'}
        solver for the different PCA methods. Please note that some solvers may not be compatible with
        some PCA methods. See PCA, TruncatedSVD and IncrementalPCA from sklearn.decompostion or
        scipy.sparse.linalg.svds.
        
    chunked : boolean
        if True, perform an incremental PCA on segments of chunk_size. The incremental PCA automatically 
        zero centers and ignores settings of random_seed and svd_solver.
        
    chunk_size : int
        Number of observations to include in each chunk. Required if chunked=True was passed.
        
    zero_center : boolean
        If True, compute standard PCA from covariance matrix. If False, omit zero-centering variables
        (uses TruncatedSVD), which allows to handle sparse input efficiently.
        
    random_state : int, RandomState, optional
        Change to use different initial states for the optimization. The default is None.

    Returns
    -------
    X_w : 2D ndarray, shape (n_samples , n_components)

    mean : 1D array, shape (n_features,)
    """
    random_state = check_random_state(random_state)

    if n_components is None:
        n_components = min(X.shape)

    if chunked:

        pca = IncrementalPCA(
            n_components=n_components, whiten=True, batch_size=chunk_size
        )
        X_w = pca.fit_transform(X)
        mean = pca.mean_

    elif issparse(X):

        if not zero_center:

            warnings.warn(
                "TruncatedSVD is very similar to PCA, but differs in that the matrix is not centered first."
                " The following components still often resemble the exact PCA very closely"
            )

            pca = TruncatedSVD(
                n_components=n_components,
                random_state=random_state,
                algorithm=svd_solver,
            )
            X_w = pca.fit_transform(X)
            X_w = (X_w / pca.singular_values_) * np.sqrt(X.shape[0] - 1)
            X_w -= X_w.mean(axis=0)
            mean = None

        else:
            X_w, mean = _pca_with_sparse(
                X, n_components, solver=svd_solver, random_state=random_state
            )

    else:

        pca = PCA(n_components=n_components, whiten=True, svd_solver=svd_solver)
        X_w = pca.fit_transform(X)
        mean = pca.mean_

    return X_w, mean


def _pca_with_sparse(
        X: spmatrix,
        npcs: int,
        solver: Optional[str] = "arpack",
        mu=None,
        random_state: Optional[Union[int, np.random.RandomState]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """ Compute PCA decomposition with initial centering for sparse input.
    
    Parameters
    ----------
    X : spmatrix, shape (n_samples, n_features)

    npcs : int
        number of PCA componnents.
        
    solver : str, optional
        Eigenvalue solver to use. Should be ‘arpack’ or ‘lobpcg’. See scipy.sparse.linalg.svds.
        The default is 'arpack'.
        
    mu : TYPE, optional
        DESCRIPTION. The default is None.
        
    random_state : int, RandomState, optional
        The default is None.

    Returns
    -------
    X_pca : 2D ndarray, shape (n_samples , n_components)

    mu : 1D array, shape (n_features,)
    """

    random_state = check_random_state(random_state)
    np.random.set_state(random_state.get_state())
    random_init = np.random.rand(np.min(X.shape))
    X = check_array(X, accept_sparse=["csr", "csc"])

    if mu is None:
        mu = X.mean(0).A.flatten()[None, :]

    # Build the linear operator that will be needed for applying svd
    mdot = mu.dot
    mmat = mdot
    mhdot = mu.T.dot
    mhmat = mu.T.dot
    Xdot = X.dot
    Xmat = Xdot
    XHdot = X.T.conj().dot
    XHmat = XHdot
    ones = np.ones(X.shape[0])[None, :].dot

    def matvec(x):
        return Xdot(x) - mdot(x)

    def matmat(x):
        return Xmat(x) - mmat(x)

    def rmatvec(x):
        return XHdot(x) - mhdot(ones(x))

    def rmatmat(x):
        return XHmat(x) - mhmat(ones(x))

    XL = LinearOperator(
        matvec=matvec,
        dtype=X.dtype,
        matmat=matmat,
        shape=X.shape,
        rmatvec=rmatvec,
        rmatmat=rmatmat,
    )

    # Apply svd
    u, s, v = svds(XL, solver=solver, k=npcs, v0=random_init)
    u, v = svd_flip(u, v)
    idx = np.argsort(-s)

    # Compute whitened projection (unit-variance and zero mean)
    X_pca = u[:, idx] * np.sqrt(u.shape[0] - 1)

    return X_pca, mu
