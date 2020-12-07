import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA  #, PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.utils import as_float_array
from sklearn import manifold
import umap
from tqdm.notebook import tqdm
from joblib import Parallel, delayed

from ._whitening import whitening

"""
Introduction
------------

Given an input matrix X (n_observations x n_variables) the ICA algorithm decomposes X into :
    X = S*A  with S (n_observations x n_variables) and A (n_variables x n_variables)
where the columns of S are independent.

In practice, we are often interested in dimension reduction and we can thus set a reduced number 
of components such that :
    X ~ S*A with S (n_observations x n_components) and A (n_components x n_variables)
where the columns of S are independent.

In signal processing the problem of blind source separation can be solved by ICA with and input
matrix X (observations = times of registration x variables = initial signals).

# In gene expression study, we typically want to unravel independent biolgical sources. To do so,
# we can apply ICA on a matrix X (observations = genes x variables = biological samples).

Important note
--------------
PLEASE NOTE THAT IN THE FOLLOWING THE MATRICES S AND A WILL BE GIVEN IN THEIR TRANSPOSED SHAPE
SIMPLY BY CONVENTION.
"""

def _ICA_decomposition(X , fun , algorithm , max_iter):
    """ Apply FastICA algorithm from sklearn.decompostion to the matrix X
               
    Parameters
    ----------
    X : 2D array-like, shape (n_observations , n_components) 
        Whitened matrix.
        
    fun : string or function.
        The functional form of the G function used in the approximation to neg-entropy. Could be either ‘logcosh’, ‘exp’
        , or ‘cube’. You can also provide your own function. It should return a tuple containing the value of the function, 
        and of its derivative, in the point.
        
    algorithm : 'parallel' or 'deflation'
        Apply parallel or deflational algorithm for FastICA
        
    max_iter : int
        see https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FastICA.html

    Returns
    -------
    2D array , shape (n_components , n_observations)
        components obtained from the ICA decomposition of X

    """
    ica = FastICA(fun = fun , algorithm = algorithm , max_iter = max_iter , whiten = False)
    return ica.fit_transform(X).T


def _centrotype(X , Sim , cluster_labels):
    """Compute the centrotype of the cluster of ICA components defined by cluster_labels
    
       centrotype : component of the cluster which is the most similar to the other components
                   of the cluster
    Parameters
    ----------
    X : 2D array, shape (n_components , n_observations)
        matrix of independent ICA components
        
    Sim : 2D array, shape (n_components , n_components)
        similarity matrix for ICA components (i.e rows of X)
        
    cluster_labels : list of integers
        indexes of the cluster of components (ex:[0 , 1 , 7] refers to the rows 0, 1 and 7 of X)

    Returns
    -------
    1D array, shape (n_observations)
        centrotype of the cluster of ICA components defined by cluster_labels

    """
    temp = np.argmax(np.sum(Sim[np.ix_(cluster_labels , cluster_labels)] , axis=0))
    return X[cluster_labels[temp] , :]

def _stability_index(Sim , cluster_labels):
    """Compute the stability index for the cluster of ICA components defined by cluster_labels.
        
       Note : Please refer to https://bmcgenomics.biomedcentral.com/track/pdf/10.1186/s12864-017-4112-9
             (section "Method") for the exact formula of the stability index.

    Parameters
    ----------
    Sim : 2D array, shape (n_components , n_components)
        similarity matrix for ICA components 
        
    cluster_labels : list of integers
        indexes of the cluster of components (ex:[0 , 1 , 7] refers to the rows 0, 1 and 7 of X)

    Returns
    -------
    Float between 0 and 1
        stability index for the cluster of ICA components defined by cluster_labels

    """
    temp = Sim[np.ix_(cluster_labels , cluster_labels)]
    ex_cluster = list(set(range(Sim.shape[1])) - set(cluster_labels))
    
    #aics = average intra-cluster similarities
    aics = (1/len(cluster_labels)**2)*np.sum(temp)  
    
    #aecs = average extra-cluster similarities
    aecs = (1/(len(ex_cluster)*len(cluster_labels)))*np.sum(Sim[np.ix_(cluster_labels , ex_cluster)]) 
    
    return aics - aecs

class StabilizedICA(object):
    """ Implement a stabilized version of the Independent Component Analysis algorithm
    
    Parameters
    ----------
    n_components : int
        number of ICA components
    
    max_iter : int
        maximum number of iteration for the FastICA algorithm
    
    n_jobs : int
        number of jobs to run in parallel. -1 means using all processors.
        See the joblib package documentation for more explanations. Default is 1.
    
    verbose: int
        control the verbosity: the higher, the more messages. Default is 0.
    
    Attributes
    ----------

    S_: 2D array, shape (n_components , n_observations)
        array of sources/metagenes, each line corresponds to a stabilized ICA component (i.e the centrotype of
        a cluster of components)   
        
    A_: 2D array, shape (n_variables , n_components)
        pseudo-inverse of S_, each column corresponds to a metasample
    
    stability_indexes_ : 1D array, shape (n_components)
        stability indexes for the stabilized ICA components
        
    Notes
    ----------
    
    n_runs is the number of time we repeat the ICA decompostion; see fit method
        
    """
    
    def __init__(self , n_components , max_iter , n_jobs = 1 , verbose = 0):
        
        self.n_components = n_components
        self.max_iter = max_iter
        self.n_jobs = n_jobs
        self.verbose = verbose
        
        self.S_ = None
        self.A_ = None
        self.stability_indexes_ = None 

    def fit(self, X ,  n_runs , fun = 'logcosh' , algorithm = 'parallel' , plot = False , normalize = True , reorientation = True , whiten = True
            , pca_solver = 'full' , chunked = False , chunk_size = None , zero_center = True):
        """1. Compute the ICA components of X n_runs times
           2. Cluster all the components (N = self.n_components*n_runs) with agglomerative 
              hierarchical clustering (average linkage) into self.n_components clusters
           3. For each cluster compute its stability index and return its centrotype as the
              final ICA component
              
           Note : Please refer to ICASSO method for more details about the process
                 (see https://www.cs.helsinki.fi/u/ahyvarin/papers/Himberg03.pdf)
                 
        Parameters
        ----------
        X : 2D array-like, shape (n_observations , n_variables) or (n_observations , n_components) if whiten is False.
            
        n_runs : int
            number of times we run the FastICA algorithm
        
        fun : string or function, optional.
            The functional form of the G function used in the approximation to neg-entropy. Could be either ‘logcosh’, ‘exp’
            , or ‘cube’. You can also provide your own function. It should return a tuple containing the value of the function, 
            and of its derivative, in the point.
            The default is 'logcosh'.
            
        algorithm : 'parallel' or 'deflation', optional.
            Apply parallel or deflational algorithm for FastICA
            The default is 'parallel'.
            
        plot : boolean, optional
            if True plot the stability indexes for each cluster in decreasing order. 
            The default is False.
        
        normalize : boolean, optional
            if True normalize the rows of S_ (i.e the stabilized ICA components) to unit standard deviation.
            The default is True.
            
        reorientation : boolean,optional
            if True re-oriente the rows of S_ towards positive heavy tail.
            The default is True.
        
        whiten : boolean, optional
            if True the matrix X is whitened, i.e centered then projected in the space defined by its 
            first self.n_components PCA components and reduced to unit variance along each of these axis. 
            If False the input X matrix must be already whitened.
            The default is True.
            
        pca_solver : str {‘auto’, ‘full’, ‘arpack’, ‘randomized’ , 'lobpcg'}, optional
            solver for the different PCA methods. Please note that some solvers may not be compatible with
            some of the PCA methods. See _whitening.py for more details.
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
        None.
        
        Note
        ------        
        If whiten is False, we suppose that X results from a whitening pre-processing step. The columns must be
        centered, scaled to unit variance and uncorrelated.
        """
        ## Initialisation
        n_observations , n_variables = X.shape
        Centrotypes = np.zeros((self.n_components , n_observations))
        Index = np.zeros(self.n_components)
        
        ## Pre-processing (whitening)
        if whiten :
            # pca = PCA(n_components = self.n_components , whiten=True , svd_solver = pca_solver)
            # X_w = pca.fit_transform(X)
            X_w = whitening(X , n_components = self.n_components , svd_solver = pca_solver , chunked = chunked , chunk_size = chunk_size
                            , zero_center = zero_center)
        else :
            X_w = as_float_array(X, copy=False)  
            
        ## Compute the self.n_components*n_runs ICA components and store into array Components
        parallel = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)
        
        maxtrials = 10
        for i in range(maxtrials):
            try:
                decomposition = parallel(delayed(_ICA_decomposition)(X_w , fun , algorithm , self.max_iter)
                                     for _ in range(n_runs))
            except ValueError:
                if i < maxtrials - 1:
                    print("FastICA from sklearn did not converge due to numerical instabilities - Retrying...")
                    continue
                else:
                    print("Too many attempts : FastICA did not converge !")
                    raise
            break
        
        self._Components = np.vstack(decomposition)
                
        ## Compute Similarity matrix between ICA components (Pearson correlation)
        self._Sim = np.abs(np.corrcoef(x=self._Components , rowvar=True))
        
        ## Cluster the components with hierarchical clustering
        clustering = AgglomerativeClustering(n_clusters = self.n_components , affinity = "precomputed" 
                            ,linkage = 'average' ).fit(1 - self._Sim)
        self._clusters = clustering.labels_ 
        
        ## For each cluster compute the stability index and the centrotype
        for i in range(self.n_components):
            cluster_labels = list(np.argwhere(clustering.labels_ == i ).flatten())
            Centrotypes[i , :] = _centrotype(self._Components , self._Sim , cluster_labels)
            Index[i] = _stability_index(self._Sim , cluster_labels)
        
        ## Sort the centrotypes (i.e final components) by stability index
        indices = np.argsort(-1*Index)
        Centrotypes , Index = Centrotypes[indices , :] , Index[indices]
        
        # Re-oriente the stabilized ICA components towards positive heaviest tails
        if reorientation : 
            self.S_ = (np.where(stats.skew(Centrotypes , axis = 1) >= 0 , 1 , -1).reshape(-1 , 1))*Centrotypes
        else : 
            self.S_ = Centrotypes
            
        # Normalize the stabilized ICA components to unit variance 
        if normalize :
            self.S_ = self.S_/(np.std(self.S_ , axis = 1).reshape(-1 ,1))
        
        self.stability_indexes_ = Index
        
        self.A_ = (X.T).dot(np.linalg.pinv(self.S_))
        #self.A_ = np.dot(X.T , np.linalg.pinv(self.S_))
        
        if plot:
            plt.figure(figsize=(10 , 7))
            plt.plot(range(1 , self.n_components + 1) , self.stability_indexes_ , linestyle='--', marker='o')
            plt.title("Stability of ICA components")
            plt.xlabel("ICA components")
            plt.ylabel("Stability index")
            
        return 
    
    def projection(self , method = "mds" , ax = None):
        """Plot the ICA components computed during fit() (N = self.n_components*n_runs) in 2D.
           Approximate the original dissimilarities between components by Euclidean distance.
           Each cluster is represented with a different color.
           
           Note : We use the dissimilarity measure sqrt(1 - |rho_ij|) (rho the Pearson correlation)
                 instead of 1 - |rho_ij| to reduce overlapping.
        
        Parameters
        ----------
        
        method : string, optional
            name of the dimensionality reduction method (e.g "tsne" , "mds" or "umap")
            The default is "umap".
            
        ax : matplotlib.axes, optional
            The default is None.
            
        Returns
        -------
        None.
        
        Note
        -------
        
        Please note that multidimensional scaling (MDS) is more computationally demanding than t-SNE or UMAP.
        However it takes into account the global structures of the data set while the others don't. For t-SNE or
        UMAP one cannot really interpret the inter-cluster distances.
        
        For more details about the UMAP (Uniform Manifold Approximation and Projection), 
        see https://pypi.org/project/umap-learn/

        """
        if method == "tsne":
            embedding = manifold.TSNE(n_components = 2 , metric = "precomputed")
        elif method == "mds" :
            embedding = manifold.MDS(n_components=2, dissimilarity= "precomputed" , n_jobs = -1)
        elif method == "umap" : 
            embedding = umap.UMAP(n_components = 2 , metric = "precomputed" )       
            
        P = embedding.fit_transform(np.sqrt(1 - self._Sim))
        
        if ax is None:
            plt.scatter(P[: , 0] , P[: , 1] , c=self._clusters , cmap = 'viridis')
        else:
            ax.scatter(P[: , 0] , P[: , 1] , c=self._clusters , cmap = 'viridis')
        return
    
    
    
def MSTD(X , m , M , step , n_runs , whiten = True , max_iter = 2000 , n_jobs = -1):
    """Plot "MSTD graphs" to help choosing an optimal dimension for ICA decomposition
        
       Run stabilized ICA algorithm for several dimensions in [m , M] and compute the
       stability distribution of the components each time
       
       Note : Please refer to https://bmcgenomics.biomedcentral.com/track/pdf/10.1186/s12864-017-4112-9
             for more details.

    Parameters
    ----------
    X : 2D array, shape (n_observations , n_variables)
    
    m : int
        minimal dimension for ICA decomposition
        
    M : int > m
        maximal dimension for ICA decomposition
        
    step : int > 0
        step between two dimensions (ex: if step = 2 the function will test the dimensions
        m, m+2, m+4, ... , M)
        
    n_runs : int
        number of times we run the FastICA algorithm (see fit method of class Stabilized_ICA)
            
    max_iter : TYPE, optional
        parameter for _ICA_decomposition. The default is 2000.
    
    n_jobs : int
        number of jobs to run in parallel for each stabilized ICA step. Default is -1

    Returns
    -------
    None.

    """
    fig, ax = plt.subplots(1 , 2 , figsize = (20 , 7))
    mean = []
    
    if whiten:
        X_w = whitening(X , n_components = M , svd_solver = 'full' , chunked = False , chunk_size = None , zero_center = True)
    
    for i in tqdm(range(m , M+step , step)):
    #for i in range(m , M+step , step): #uncomment if you don't want to use tqdm (and comment the line above !)
        s = StabilizedICA(i , max_iter ,n_jobs)
        s.fit(X_w[: , :m] , n_runs , whiten = False)
        mean.append(np.mean(s.stability_indexes_))
        ax[0].plot(range(1 , len(s.stability_indexes_)+1) , s.stability_indexes_ , 'k')
        
    ax[1].plot(range(m , M+step , step) , mean) 
    
    ax[1].set_title("Mean stability")
    ax[1].set_xlabel("Number of components")
    ax[0].set_title("Index stability distribution")
    ax[0].set_xlabel("Number of components")    
    return 