from anndata import AnnData
from .base import StabilizedICA
import warnings


def ica(
    data,
    observations,
    n_components,
    n_runs,
    resampling=None,
    return_info=False,
    copy=False,
    plot_projection=None,
    fun="logcosh",
    algorithm="fastica_par",
    normalize=True,
    reorientation=True,
    whiten=True,
    pca_solver="auto",
    chunked=False,
    chunk_size=None,
    zero_center=True,
):
    """ Compute stabilized ICA decomposition for AnnData formats. Use the implementation of stabilized ICA in
    the same package (see module sica.base.py)
    
    Parameters
    ----------
    data : AnnData, ndarray, spmatrix, shape (n_cells , n_genes)
        The (annotated) data matrix.
        
    observations : str {'genes' , 'cells'}
        This parameter allows the user to choose which of the metagenes or the metasamples he wants to consider as ICA independent
        sources.
        
        - If ``observations = 'genes'`` the independent sources will be of shape (n_genes). The metagenes will correspond to the independent ICA sources while the metasamples will correspond to the linear mixing.
        - If ``observations = 'cells'`` the independent sources will be of shape (n_cells). The metasamples will correspond to the independent ICA sources while the metagenes will correspond to the linear mixing.
        
    n_components : int
        Number of stabilized ICA components.
        
    n_runs : int
        Number of times we repeat the FastICA algorithm.
    
    resampling : str {None , 'bootstrap' , 'fast_bootstrap'}, optional
        Method for resampling the data before each run of the ICA solver.
        
        - If None, no resampling is applied.
        - If 'bootstrap' the classical bootstrap method is applied to the original data matrix, the resampled matrix is whitened (using the whitening hyperparameters set for the fit method) and the ICA components are extracted.
        - If 'fast_boostrap' a fast bootstrap algorithm is applied to the original data matrix and the whitening operation is performed simultaneously with SVD decomposition and then the ICA components are extracted (see References).
        
        Resampling could lead to quite heavy computations (whitening at each iteration), depending on the size of the input data. It should be considered with care. The default is None.
        
    return_info : bool, optionnal
        See results. The default is false.
        
    copy : bool, optionnal
        See results. The default is false.
        
    plot_projection : str {'mds' ,'tsne' , 'umap'}, optional
        Name of the dimensionality reduction method. If ``None``, this projection is
        not computed.
        The default is None.
    
    fun : str {'cube' , 'exp' , 'logcosh' , 'tanh'} or function, optional.

        If ``algorithm`` is in {'fastica_par' , 'fastica_def'}, it represents the functional form of the G function used
        in the approximation to neg-entropy. Could be either ‘logcosh’, ‘exp’, or ‘cube’.

        If ``algorithm`` is in {'fastica_picard' , 'infomax' , 'infomax_ext' , 'infomax_orth'}, it is associated with
        the choice of a density model for the sources. See supplementary explanations for more details.

        The default is 'logcosh'.

    algorithm : str {'fastica_par' , 'fastica_def' , 'fastica_picard' , 'infomax' , 'infomax_ext' , 'infomax_orth'}, optional.
            The algorithm applied for solving the ICA problem at each run. Please see the supplementary explanations for
            more details.
            The default is 'fastica_par', i.e. FastICA from sklearn with parallel implementation.

    normalize : boolean, optional
        If True normalize the rows of ``S_`` (i.e. the stabilized ICA components) to unit standard deviation.
        The default is True.

    reorientation : boolean,optional
        If True re-oriente the rows of ``S_`` towards positive heavy tail.
        The default is True.

    whiten : boolean, optional

        If True the matrix X is whitened, i.e. centered then projected in the space defined by its
        first ``n_components`` PCA components and reduced to unit variance along each of these axes.

        If False the input X matrix must be already whitened (the columns must be centered, scaled to unit
        variance and uncorrelated.)

        The default is True.

    pca_solver : str {‘auto’, ‘full’, ‘arpack’, ‘randomized’ , 'lobpcg'}, optional
        Solver for the different PCA methods. Please note that some solvers may not be compatible with
        some PCA methods. See _whitening.py for more details.
        The default is "full" (i.e SVD decomposition)

    chunked : boolean, optional
        Parameter for the whitening step, see _whitening.py for more details.
        The default is False.

    chunk_size : int, optional
        Parameter for the whitening step, see _whitening.py for more details.
        The default is None.

    zero_center : boolean, optional
        Parameter for the whitening step, see _whitening.py for more details.
        The default is True.
    
    Returns
    -------
    Metasamples : 2D array, shape (n_cells ,  n_components)
        If ``data`` is array-like and ``return_info = False`` this function only returns the metasamples.  
        
        If ``observations = 'genes'`` it corresponds to the mixing matrix.   
        
        If ``observations = 'cells'`` it corresponds to the independent sources.
        
    Metagenes :2D array, shape (n_components , n_genes)
        If ``data`` is array-like and ``return_info = True`` this function returns the metagenes.   
        
        If ``observations = 'genes'`` it corresponds to the independent sources.   
        
        If ``observations = 'cells'`` it corresponds to the mixing matrix.
        
    stability_indexes : 1D array, shape (n_components)
        If ``data`` is array-like and ``return_info = True`` this function returns the stability indexes for the stabilized ICA 
        components.
        
    adata : AnnData
        …otherwise if copy=True it returns or else adds fields to ``data``:           
           .obsm['sica_metasamples'] 
           
           .varm['sica_metagenes'] 
           
           .uns['sica']['stability_indexes']  
        
    Examples
    --------
    >>> import scanpy
    >>> from sica.singlecell import ica    
    >>> anndata = scanpy.read_h5ad('GSE90860_3.h5ad')
    >>> anndata.X -= anndata.X.mean(axis =0)
    >>> ica(anndata , observations = 'genes' , n_components = 30 , n_runs = 100)

    """

    #### 0. Initialisation

    data_is_AnnData = isinstance(data, AnnData)
    if data_is_AnnData:
        adata = data.copy() if copy else data
    else:
        adata = AnnData(data)

    X = adata.X
    sica = StabilizedICA(
            n_components=n_components,
            n_runs=n_runs,
            resampling=resampling,
            algorithm=algorithm,
            fun=fun,
            whiten=whiten,
            normalize=normalize,
            reorientation=reorientation,
            pca_solver=pca_solver,
            chunked=chunked,
            chunk_size=chunk_size,
            zero_center=zero_center,
            n_jobs=-1,
    )

    #### 1. Apply stabilized ICA

    # Apply stabilized ICA to discover independent metagenes
    if observations == "genes":

        sica.fit(X.T)

    # Apply stabilized ICA to discover independent metasamples
    elif observations == "cells":

        # If no resampling is needed and 'X_pca' is already computed, we try to use it
        if (resampling is None) and whiten and "X_pca" in adata.obsm:

            n_pcs = adata.obsm["X_pca"].shape[1]

            if (n_components is None and n_pcs < min(X.shape)) or (
                n_components > n_pcs
            ):

                warnings.warn(
                    "The number of PCA components in adata.obsm['X_pca'] is strictly less than n_components."
                    "By default, the PCA step is redone within the stabilized ICA algorithm with the desired number "
                    "of components (i.e n_components). "
                )

                sica.fit(X)

            else:
                sica.set_params({"whiten": False})
                sica.fit(adata.obsm["X_pca"][:, :n_components])
        else:
            sica.fit(X)

    #### 2. Plot 2D projection (optional)

    if plot_projection is not None:
        sica.projection(method=plot_projection)

    #### 3. Return data

    if observations == "genes":
        metasamples = sica.transform(X.T)
        metagenes = sica.S_
    elif observations == "cells":
        metasamples = sica.S_.T
        metagenes = sica.transform(X).T
    stability_indexes = sica.stability_indexes_

    if data_is_AnnData:

        adata.obsm["sica_metasamples"] = metasamples
        adata.varm["sica_metagenes"] = metagenes.T
        adata.uns["sica"] = {}
        adata.uns["sica"]["stability_indexes"] = stability_indexes

        return adata if copy else None

    else:
        if return_info:
            return metasamples, metagenes, stability_indexes
        else:
            return metasamples
