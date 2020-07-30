import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA
from sklearn.cluster import AgglomerativeClustering
from sklearn import manifold
from tqdm.notebook import tqdm

def _ICA_decomposition(X , n_components, max_iter):
    """ Apply FastICA algorithm from sklearn.decompostion to the matrix X
        
        Note: FastICA in sklearn works with a matrix of shape (n_features , n_samples)
              that is why we fit FastICA with X.T
              
    Parameters
    ----------
    X : 2D array, shape (n_samples , n_features)

    n_components : int
        number of ICA components
        
    max_iter : int
        see https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FastICA.html

    Returns
    -------
    2D array , shape (n_components , n_features)
        components obtained from the ICA decomposition of X

    """
    ica = FastICA(n_components = n_components , max_iter = max_iter)
    ica.fit(X.T)
    return ica.transform(X.T).T

def _centrotype(X , S , cluster_labels):
    """Compute the centrotype of the cluster of ICA components defined by cluster_labels
    
       centrotype: component of the cluster which is the most similar to the other components
                   of the cluster
    Parameters
    ----------
    X : 2D array, shape (n_components , n_features)
        matrix of independent ICA components
        
    S : 2D array, shape (n_components , n_components)
        similarity matrix for ICA components (i.e rows of X)
        
    cluster_labels : list of integers
        indexes of the cluster of components (ex:[0 , 1 , 7] refers to the rows 0, 1 and 7 of X)

    Returns
    -------
    1D array, shape ( , n_features)
        centrotype of the cluster of ICA components defined by cluster_labels

    """
    temp = np.argmax(np.sum(S[np.ix_(cluster_labels , cluster_labels)] , axis=0))
    return X[cluster_labels[temp] , :]

def _stability_index(S , cluster_labels):
    """Compute the stability index for the cluster of ICA components defined by cluster_labels.
        
       Note: Please refer to https://bmcgenomics.biomedcentral.com/track/pdf/10.1186/s12864-017-4112-9
             (section "Method") for the exact formula of the stability index.

    Parameters
    ----------
    S : 2D array, shape (n_components , n_components)
        similarity matrix for ICA components 
        
    cluster_labels : list of integers
        indexes of the cluster of components (ex:[0 , 1 , 7] refers to the rows 0, 1 and 7 of X)

    Returns
    -------
    Float between 0 and 1
        stability index for the cluster of ICA components defined by cluster_labels

    """
    temp = S[np.ix_(cluster_labels , cluster_labels)]
    ex_cluster = list(set(range(S.shape[1])) - set(cluster_labels))
    
    #aics = average intra-cluster similarities
    aics = (1/len(cluster_labels)**2)*np.sum(temp)  
    
    #aecs = average extra-cluster similarities
    aecs = (1/(len(ex_cluster)*len(cluster_labels)))*np.sum(S[np.ix_(cluster_labels , ex_cluster)]) 
    
    return aics - aecs

class StabilizedICA(object):
    """ Implement a stabilized version of the Independent Component Analysis algorithm
    
    Parameters
    ----------
    n_components: int
        number of ICA components
    
    max_iter: int
        maximum number of iteration for the FastICA algorithm
    
    Attributes
    ----------
    S: numpy array, shape (n_runs*n_components , n_runs*n_components)
        Similarity matrix between all ICA components (absolute values of Pearson correlations)
        
    clusters: numpy array, shape (n_runs*n_samples)
        cluster labels for each ICA component
        
    Notes
    ----------
    
    n_runs is the number of time we repeat the ICA decompostion; see fit method
        
    """
    
    def __init__(self , n_components , max_iter):
        
        self.n_components = n_components
        self.max_iter = max_iter
        
        self.S = None  #Similarity matrix between all ICA components
        self.clusters = None
    
    def fit(self, X ,  n_runs , plot = False):
        """1. Compute the ICA components of X n_runs times
           2. Cluster all the components (N = self.n_components*n_runs) with agglomerative 
              hierarchical clustering (average linkage) into self.n_components clusters
           3. For each cluster compute its stability index and return its centrotype as the
              final ICA component
              
           Note: Please refer to ICASSO method for more details about the process
                 (see https://www.cs.helsinki.fi/u/ahyvarin/papers/Himberg03.pdf)
                 
        Parameters
        ----------
        X : 2D array, shape (n_samples , n_features)
            
        n_runs : int
            number of times we run the FastICA algorithm
            
        plot : boolean, optional
            if True plot the stability indexes for each cluster in decreasing order. 
            The default is False.

        Returns
        -------
        Index : 1D array, shape (n_components)
            stability indexes for the n_components clusters
            
        Centrotypes : 2D array, shape (n_components, n_features)
            final ICA components (i.e centrotypes of each cluster)

        """
        ## Initialisation
        n_samples , n_features = X.shape
        Components = np.zeros((n_runs*self.n_components , n_features))
        Centrotypes = np.zeros((self.n_components , n_features))
        Index = np.zeros(self.n_components)
        
        ## Compute the self.n_components*n_runs ICA components and store into array Components
        for i in range(n_runs):
            Components[i*self.n_components : (i+1)*self.n_components , : ] = _ICA_decomposition(X , self.n_components , self.max_iter)
        
        ## Compute Similarity matrix between ICA components (Pearson correlation)
        self.S = np.abs(np.corrcoef(x=Components , rowvar=True))
        
        ## Cluster the components with hierarchical clustering
        clustering = AgglomerativeClustering(n_clusters = self.n_components , affinity = "precomputed" 
                            ,linkage = 'average' ).fit(1 - self.S)
        self.clusters = clustering.labels_ 
        
        ## For each cluster compute the stability index and the centrotype
        for i in range(self.n_components):
            cluster_labels = list(np.argwhere(clustering.labels_ == i ).flatten())
            Centrotypes[i , :] = _centrotype(Components , self.S , cluster_labels)
            Index[i] = _stability_index(self.S , cluster_labels)
        
        ## Sort the centrotypes (i.e final components) by stability index
        indices = np.argsort(-1*Index)
        Centrotypes , Index = Centrotypes[indices , :] , Index[indices]
        
        if plot:
            plt.figure(figsize=(10 , 7))
            plt.plot(range(1 , self.n_components + 1) , Index , linestyle='--', marker='o')
            plt.title("Stability of ICA components")
            plt.xlabel("ICA components")
            plt.ylabel("Stability index")
            
        return Index, Centrotypes
    
    def projection(self , ax = None):
        """Plot the ICA components computed during fit() (N = self.n_components*n_runs) in 2D.
           Approximate the original dissimilarities between components by Euclidean distance.
           Each cluster is represented with a different color.
           
           Note: We use the dissimilarity measure sqrt(1 - |rho_ij|) (rho the Pearson correlation)
                 instead of 1 - |rho_ij| to reduce overlapping.
        
        Parameters
        ----------
        
        ax : matplotlib.axes
            
        Returns
        -------
        None.

        """
        embedding = manifold.MDS(n_components=2, dissimilarity= "precomputed", random_state=6)
        P = embedding.fit_transform(np.sqrt(1 - self.S))
        if ax is None:
            plt.scatter(P[: , 0] , P[: , 1] , c=self.clusters , cmap = 'viridis')
            plt.title("")
        else:
            ax.scatter(P[: , 0] , P[: , 1] , c=self.clusters , cmap = 'viridis')
        return
    
    def metasamples(self, X , Centrotypes):
        """Compute the meta-samples with the formula A = XS^{+} (where S^{+} is the pseudo-inverse
           of the matrix of metagenes S)
        
        Parameters
        ----------
        X : 2D array, shape (n_samples , n_features)

        Centrotypes : 2D array, shape (n_components, n_features)
            final ICA components (i.e centrotypes of each cluster)

        Returns
        -------
        2D array, shape (n_samples , n_components)
            matrix of meta-samples (i.e A so that X = AS)

        """
        return np.dot(X , np.linalg.pinv(Centrotypes))
    
    def clear(self):
        self.clusters , self.S = None , None
        return 
    
    
def MSTD(X , m , M , step , n_runs , max_iter = 2000):
    """Plot "MSTD graphs" to help choosing an optimal dimension for ICA decomposition
        
       Run stabilized ICA algorithm for several dimensions in [m , M] and compute the
       stability distribution of the components each time
       
       Note: Please refer to https://bmcgenomics.biomedcentral.com/track/pdf/10.1186/s12864-017-4112-9
             for more details.

    Parameters
    ----------
    X : 2D array, shape (n_samples , n_features)
    
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

    Returns
    -------
    None.

    """
    fig, ax = plt.subplots(1 , 2 , figsize = (20 , 7))
    mean = []
    for i in tqdm(range(m , M+step , step)):
    #for i in range(m , M+step , step): #uncomment if you don't want to use tqdm (and comment the line above !)
        s = StabilizedICA(i , max_iter)
        Index,*_ = s.fit(X , n_runs)
        mean.append(np.mean(Index))
        ax[0].plot(range(1 , len(Index)+1) , Index , 'k')
        
    ax[1].plot(range(m , M+step , step) , mean) 
    
    ax[1].set_title("Mean stability")
    ax[1].set_xlabel("Number of components")
    ax[0].set_title("Index stability distribution")
    ax[0].set_xlabel("Number of components")    
    return 