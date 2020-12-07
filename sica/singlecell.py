from anndata import AnnData
from .base import StabilizedICA
import warnings

def ica(data, observations , n_components,  n_runs , return_info = False , copy = False , plot_projection = None , fun = 'logcosh' , algorithm = 'parallel' 
     , normalize = True , reorientation = True , whiten = True , pca_solver = 'full', chunked = False , chunk_size = None , zero_center = True):
    """ Compute stabilized ICA decomposition for AnnData formats. Use the implementation of stabilized ICA in
    the same package (see module base.py)
    
    Parameters
    ----------
    data : AnnData, ndarray, spmatrix, shape (n_cells , n_genes)
        The (annotated) data matrix.
        
    observations : str {'genes' , 'cells'}
        If observations == 'genes' the independent sources will be of shape (n_genes)
        else they will be of shape (n_cells). This parameter allows the user to choose
        which of the metagenes or the metasamples he wants to consider as ICA independent
        sources.
        
    n_components : int
        number of stabilized ICA components.
        
    n_runs : int
        number of times we repeat the FastICA algorithm
        
    return_info : bool, optionnal
        See results. The default is false.
        
    copy : bool, optionnal
        See resutls. The default is false.
        
    plot_projection : str {'mds' ,'tsne' , 'umap'}, optional
        name of the dimensionality reduction method. If None, this projection is
        not computed.
        The default is None.
    
    Returns
    -------
    spmatrix, ndarray
    If data is array-like and return_info=False was passed, this function only returns projection
    in the ICA base. If return_info = True, stability indexes and independent sources are also returned.
    
    AnnData
    …otherwise if copy=True it returns or else adds fields to adata:
    .obsm['sica_metasamples']
    .varm['sica_metagenes']
    .uns['sica']['stability_indexes']

    """
    
    data_is_AnnData = isinstance(data, AnnData)
    if data_is_AnnData:
        adata = data.copy() if copy else data
    else:
        adata = AnnData(data)

    X = adata.X
    sica = StabilizedICA(n_components = n_components ,  max_iter = 2000 , n_jobs = -1)

    # Apply stabilized ICA to discover independent metagenes
    if observations == 'genes':

        sica.fit(X.T , n_runs = n_runs , fun = fun , algorithm = algorithm , normalize = normalize , reorientation = reorientation , whiten = whiten
                 ,pca_solver = pca_solver , chunked = chunked , chunk_size = chunk_size , zero_center = zero_center )
        
    # Apply stabilized ICA to discover independent metasamples
    elif observations == 'cells' :
        if whiten and 'X_pca' in adata.obsm:
            
            n_pcs = adata.obsm['X_pca'].shape[1]
            
            if (n_components is None and n_pcs < min(X.shape) ) or (n_components > n_pcs) :
                
                warnings.warn("The number of PCA components in adata.obsm['X_pca'] is strictly less than n_components."
                              " By default, the PCA step is redone within the stabilized ICA algorithm with the desired number of components (i.e n_components)." )
                
                sica.fit(X , n_runs = n_runs , fun = fun , algorithm = algorithm , normalize = normalize , reorientation = reorientation , whiten = whiten
                 ,pca_solver = pca_solver , chunked = chunked , chunk_size = chunk_size , zero_center = zero_center )
            else :
                sica.fit(adata.obsm['X_pca'][: , : n_components] , n_runs = n_runs , fun = fun , algorithm = algorithm , normalize = normalize
                         , reorientation = reorientation , whiten = False)
        else :           
            sica.fit(X , n_runs = n_runs , fun = fun , algorithm = algorithm , normalize = normalize , reorientation = reorientation , whiten = whiten
                 ,pca_solver = pca_solver , chunked = chunked , chunk_size = chunk_size , zero_center = zero_center )           
    
    # Plot 2D projection
    if plot_projection is not None:        
      sica.projection(method = plot_projection)  
        
    # Return data
    if data_is_AnnData:
        
        adata.obsm['sica_metasamples'] = sica.A_ if observations == 'genes' else sica.S_.T
        adata.varm['sica_metagenes'] = sica.S_.T if observations == 'genes' else sica.A_
        adata.uns['sica'] = {}
        adata.uns['sica']['stability_indexes'] = sica.stability_indexes_  
        
        return adata if copy else None
    
    else :
        if return_info:
            return (sica.A_ , sica.S_ , sica.stability_indexes_) if observations == 'genes' else (sica.S_.T , sica.A_.T , sica.stability_indexes_)
        else:
            return sica.A_ if observations == 'genes' else sica.S_.T
