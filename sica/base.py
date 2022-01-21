import numpy as np
import scipy.stats as stats
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib.axes
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import FastICA
from sklearn.utils import as_float_array, check_array
from sklearn.utils.validation import FLOAT_DTYPES
from sklearn.utils.extmath import svd_flip
from sklearn import manifold
import umap
from tqdm.notebook import tqdm
from joblib import Parallel, delayed
import warnings

from picard import picard
from ._whitening import whitening


def _check_algorithm(algorithm, fun):

    all_algorithms = [
        "fastica_par",
        "fastica_def",
        "fastica_picard",
        "infomax",
        "infomax_ext",
        "infomax_orth",
    ]
    if algorithm not in all_algorithms:
        raise ValueError(
            "Stabilized ICA supports only algorithms in %s, got"
            " %s." % (all_algorithms, algorithm)
        )

    all_funs = ["exp", "cube", "logcosh", "tanh"]
    if (isinstance(fun, str)) and (fun not in all_funs):
        raise ValueError(
            "Stabilized ICA supports only string functions in %s, got"
            " %s. Please see sklearn.FastICA or picard for alternatives (customed functions)"
            % (all_algorithms, algorithm)
        )

    if fun == "tanh" and algorithm in ["fastica_par", "fastica_def"]:
        warnings.warn(
            " 'tanh' is not available for sklearn.FastICA. By default, we assumed 'logcosh' was the desired function"
        )
        fun = "logcosh"

    if fun == "logcosh" and algorithm in [
        "fastica_picard",
        "infomax",
        "infomax_ext",
        "infomax_orth",
    ]:
        warnings.warn(
            "'logcosh' is not available for picard. By default, we assumed 'tanh' was the desired function"
        )
        fun = "tanh"

    if fun != "tanh" and algorithm in ["fastica_picard", "infomax_ext"]:
        warnings.warn(
            "Using a different density than `'tanh'` may lead to erratic behavior of the picard algorithm"
            " when extended=True (see picard package for more explanations)"
        )

    if fun == "exp" and algorithm == "infomax":
        warnings.warn(
            "Using the exponential density model may lead to a FloatingPointError. To solve this problem you may try to scale the non-linearity changing the alpha parameter in the exp density"
            " (ex : set the `fun` parameter of fit method to `picard.densities.Exp(params={'alpha': 0.1})`)"
        )

    if algorithm == "fastica_par":
        return "fastica", {"algorithm": "parallel", "fun": fun}
    elif algorithm == "fastica_def":
        return "fastica", {"algorithm": "deflation", "fun": fun}
    elif algorithm == "fastica_picard":
        return "picard", {"ortho": True, "extended": True, "fun": fun}
    elif algorithm == "infomax":
        return "picard", {"ortho": False, "extended": False, "fun": fun}
    elif algorithm == "infomax_ext":
        return "picard", {"ortho": False, "extended": True, "fun": fun}
    elif algorithm == "infomax_orth":
        return "picard", {"ortho": True, "extended": False, "fun": fun}


def _centrotype(X, Sim, cluster_labels):
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
    temp = np.argmax(np.sum(Sim[np.ix_(cluster_labels, cluster_labels)], axis=0))
    return X[cluster_labels[temp], :]


def _stability_index(Sim, cluster_labels):
    """Compute the stability index for the cluster of ICA components defined by cluster_labels.
        
    Please refer to https://bmcgenomics.biomedcentral.com/track/pdf/10.1186/s12864-017-4112-9
    (section "Method") for the exact formula of the stability index.

    Parameters
    ----------
    Sim : 2D array, shape (n_components , n_components)
        similarity matrix for ICA components 
        
    cluster_labels : list of integers
        indexes of the cluster of components (ex: [0 , 1 , 7] refers to the rows 0, 1 and 7 of X)

    Returns
    -------
    Float between 0 and 1
        stability index for the cluster of ICA components defined by cluster_labels

    """
    temp = Sim[np.ix_(cluster_labels, cluster_labels)]
    ex_cluster = list(set(range(Sim.shape[1])) - set(cluster_labels))

    # aics = average intra-cluster similarities
    aics = (1 / len(cluster_labels) ** 2) * np.sum(temp)

    # aecs = average extra-cluster similarities
    aecs = (1 / (len(ex_cluster) * len(cluster_labels))) * np.sum(
        Sim[np.ix_(cluster_labels, ex_cluster)]
    )

    return aics - aecs


class StabilizedICA(object):
    """ Implement a stabilized version of the Independent Component Analysis algorithm
    
    Parameters
    ----------
    n_components : int
        Number of ICA components.
    
    max_iter : int
        Maximum number of iteration for the FastICA algorithm.
    
    resampling : str {None , 'bootstrap' , 'fast_bootstrap'}, optional
        Method for resampling the data before each run of the ICA solver.
        
        - If None, no resampling is applied.
        - If 'bootstrap' the classical bootstrap method is applied to the original data matrix, the resampled matrix is whitened (using the whitening hyperparameters set for the fit method) and the ICA components are extracted.
        - If 'fast_boostrap' a fast bootstrap algorithm is applied to the original data matrix and the whitening operation is performed simultaneously with SVD decomposition and then the ICA components are extracted (see References).
        
        Resampling could lead to quite heavy computations (whitening at each iteration), depending on the size of the input data. It should be considered with care. The default is None.
        
    n_jobs : int, optional
        Number of jobs to run in parallel. -1 means using all processors.
        See the joblib package documentation for more explanations. The default is 1.
    
    verbose: int, optional
        Control the verbosity: the higher, the more messages. The default is 0.
    
    Attributes
    ----------
    S_: 2D array, shape (n_components , n_observations)
        Array of sources/metagenes, each line corresponds to a stabilized ICA component (i.e the centrotype of
        a cluster of components).  
        
    A_: 2D array, shape (n_mixtures , n_components)
        Pseudo-inverse of ``S_``, each column corresponds to a metasample.
    
    stability_indexes_ : 1D array, shape (n_components)
        Stability indexes for the stabilized ICA components.
    
    Notes
    -----
    Here `n_components` corresponds to the number of ICA sources, `n_mixtures` corresponds to the number of linear mixtures (i.e linear mixtures of ICA sources) that we observe,
    and `n_observations` corresponds to the number of observations collected for these mixtures.
    Each time, the user needs to carefully determine which dimension in his data set should correspond to the linear mixtures of ICA sources and which dimension should correspond to the observations. 
    The user should keep in mind that, at the end, he will obtain `n_components` vectors of dimension `n_observations`, independent form each other (as finite samples of latent independent distributions).
    The user guide and the definition of the ICA framework should be helpful.
    
    - For a data set of discretized sound signals registered by 10 microphones at 100 time points, if we want to retrieve 5 ICA sources we need to set `n_mixtures = 10`, `n_observations = 100` and `n_components = 5`.
    - For a gene expression data set with 100 samples and 10000 genes, if we want to retrieve 10 ICA sources **in the gene space** we need to set `n_mixtures = 100`, `n_observations = 10000` and `n_components = 10`.
    
    References
    ----------
    Fast bootstrap algorithm :
        Fisher A, Caffo B, Schwartz B, Zipunnikov V. Fast, Exact Bootstrap Principal Component Analysis for p > 1 million.
        J Am Stat Assoc. 2016;111(514):846-860. doi: 10.1080/01621459.2015.1062383. Epub 2016 Aug 18. PMID: 27616801; PMCID: PMC5014451.
        
    ICASSO method :
        J. Himberg and A. Hyvarinen, "Icasso: software for investigating the reliability of ICA estimates by clustering and visualization," 
        2003 IEEE XIII Workshop on Neural Networks for Signal Processing (IEEE Cat. No.03TH8718), 2003, pp. 259-268, doi: 10.1109/NNSP.2003.1318025
        (see https://www.cs.helsinki.fi/u/ahyvarin/papers/Himberg03.pdf). 
    
    UMAP :
        For more details about the UMAP (Uniform Manifold Approximation and Projection), see https://pypi.org/project/umap-learn/.
    
    Examples
    --------
    >>> import pandas as pd
    >>> from sica.base import StabilizedICA   
    >>> df = pd.read_csv("data.csv" , index_col = 0).transpose()  
    >>> sICA = StabilizedICA(n_components = 45 , max_iter = 2000 , n_jobs = -1)
    >>> sICA.fit(df , n_runs = 30 , plot = True , normalize = True)    
    >>> Sources = pd.DataFrame(sICA.S_ , columns = df.index , index = ['source ' + str(i) for i in range(sICA.S_.shape[0])])
    >>> Sources.head()                
    """

    def __init__(self, n_components, max_iter, resampling=None, n_jobs=1, verbose=0):

        self.n_components = n_components
        self.max_iter = max_iter
        self.resampling = resampling
        self.n_jobs = n_jobs
        self.verbose = verbose

        self.S_ = None
        self.A_ = None
        self.stability_indexes_ = None

    def fit(
        self,
        X,
        n_runs,
        fun="logcosh",
        algorithm="fastica_par",
        plot=False,
        normalize=True,
        reorientation=True,
        whiten=True,
        pca_solver="full",
        chunked=False,
        chunk_size=None,
        zero_center=True,
    ):
        """ Fit the ICA model with X (use stabilization).
        
        1. Compute the ICA components of X ``n_runs`` times.
        
        2. Cluster all the ``n_components*n_runs`` components with agglomerative 
           hierarchical clustering (average linkage) into ``n_components`` clusters.
           
        3. For each cluster compute its stability index and return its centrotype as the
           final ICA component.              
                 
        Parameters
        ----------
        X : 2D array-like, shape (n_observations , n_mixtures) or (n_observations , n_components) if whiten is False.
            Training data 
            
        n_runs : int
            Number of times we run the FastICA algorithm
        
        fun : str {'cube' , 'exp' , 'logcosh' , 'tanh'} or function, optional.
        
            If ``algorithm`` is in {'fastica_par' , 'fastica_def'}, it represents the functional form of the G function used in 
            the approximation to neg-entropy. Could be either ‘logcosh’, ‘exp’, or ‘cube’.
            
            If ``algorithm`` is in {'fastica_picard' , 'infomax' , 'infomax_ext' , 'infomax_orth'}, it is associated with the choice of
            a density model for the sources. See supplementary explanations for more details.
            
            The default is 'logcosh'.
            
        algorithm : str {'fastica_par' , 'fastica_def' , 'fastica_picard' , 'infomax' , 'infomax_ext' , 'infomax_orth'}, optional.
            The algorithm applied for solving the ICA problem at each run. Please the supplementary explanations for more details.
            The default is 'fastica_par', i.e FastICA from sklearn with parallel implementation.
              
        plot : boolean, optional
            If True plot the stability indexes for each cluster in decreasing order. 
            The default is False.
        
        normalize : boolean, optional
            If True normalize the rows of ``S_`` (i.e the stabilized ICA components) to unit standard deviation.
            The default is True.
            
        reorientation : boolean,optional
            If True re-oriente the rows of ``S_`` towards positive heavy tail.
            The default is True.
        
        whiten : boolean, optional
        
            If True the matrix X is whitened, i.e centered then projected in the space defined by its 
            first ``n_components`` PCA components and reduced to unit variance along each of these axis. 
            
            If False the input X matrix must be already whitened (the columns must be centered, scaled to unit 
            variance and uncorrelated.)
            
            The default is True.
            
        pca_solver : str {‘auto’, ‘full’, ‘arpack’, ‘randomized’ , 'lobpcg'}, optional
            Solver for the different PCA methods. Please note that some solvers may not be compatible with
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
        """
        #### 0. Initialisation

        n_observations, n_mixtures = X.shape
        Centrotypes = np.zeros((self.n_components, n_observations))
        Index = np.zeros(self.n_components)

        self._method, self._solver_params = _check_algorithm(algorithm, fun)
        X = check_array(X, dtype=FLOAT_DTYPES, accept_sparse=True, copy=whiten)

        parallel = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)

        #### 1. Compute the n_components*n_runs ICA components depending on the resampling strategy

        if self.resampling is None:

            # Pre-processing (whitening)
            if whiten:
                X_w = whitening(
                    X,
                    n_components=self.n_components,
                    svd_solver=pca_solver,
                    chunked=chunked,
                    chunk_size=chunk_size,
                    zero_center=zero_center,
                )
            else:
                X_w = as_float_array(X, copy=False)

            # Compute the n_components*n_runs ICA components
            decomposition = self._parallel_decomposition(
                parallel=parallel,
                func=self._ICA_decomposition,
                kwargs={"X_w": X_w},
                algorithm=algorithm,
                n_runs=n_runs,
            )

        elif self.resampling == "bootstrap":

            if not whiten:
                raise ValueError(
                    "The matrix X should not be pre-whitened when resampling = 'bootstrap'. The whitening step is performed consecutively to each resampling (using the whitening hyperparameters set by the user)."
                )

            # Pre-processing (save whitening parameters)
            whitening_params = {
                "svd_solver": pca_solver,
                "chunked": chunked,
                "chunk_size": chunk_size,
                "zero_center": zero_center,
            }

            # Compute the n_components*n_runs ICA components
            decomposition = self._parallel_decomposition(
                parallel=parallel,
                func=self._ICA_decomposition_bootstrap,
                kwargs={"X": X, "whitening_params": whitening_params},
                algorithm=algorithm,
                n_runs=n_runs,
            )

        elif self.resampling == "fast_bootstrap":

            if not whiten:
                raise ValueError(
                    "The matrix X should not be pre-whitened when resampling = 'fast_bootstrap'. The whitening step is performed consecutively to each resampling (with SVD decomposition)."
                )

            # Pre-processing (initial svd decomposition)
            U, S, Vt = linalg.svd(X - np.mean(X, axis=0), full_matrices=False)
            SVt = np.dot(np.diag(S), Vt)

            # Compute the n_components*n_runs ICA components
            decomposition = self._parallel_decomposition(
                parallel=parallel,
                func=self._ICA_decomposition_fast_bootstrap,
                kwargs={"U": U, "SVt": SVt},
                algorithm=algorithm,
                n_runs=n_runs,
            )

        else:
            raise ValueError(
                "Unrecognized resampling method. Please choose among None, 'bootstrap' or 'fast_bootstrap'"
            )

        self._Components = np.vstack(decomposition)

        #### 2. Cluster the n_components*n_runs ICA components with hierarchical clustering

        # Compute Similarity matrix between ICA components (Pearson correlation)
        self._Sim = np.abs(np.corrcoef(x=self._Components, rowvar=True))

        # Cluster the components with hierarchical clustering
        clustering = AgglomerativeClustering(
            n_clusters=self.n_components, affinity="precomputed", linkage="average"
        ).fit(1 - self._Sim)
        self._clusters = clustering.labels_

        #### 3. For each cluster compute the stability index and the centrotype

        for i in range(self.n_components):
            cluster_labels = list(np.argwhere(clustering.labels_ == i).flatten())
            Centrotypes[i, :] = _centrotype(self._Components, self._Sim, cluster_labels)
            Index[i] = _stability_index(self._Sim, cluster_labels)

        # Sort the centrotypes (i.e final components) by stability index
        indices = np.argsort(-1 * Index)
        Centrotypes, Index = Centrotypes[indices, :], Index[indices]

        # Re-oriente the stabilized ICA components towards positive heaviest tails
        if reorientation:
            self.S_ = (
                np.where(stats.skew(Centrotypes, axis=1) >= 0, 1, -1).reshape(-1, 1)
            ) * Centrotypes
        else:
            self.S_ = Centrotypes

        # Normalize the stabilized ICA components to unit variance
        if normalize:
            self.S_ = self.S_ / (np.std(self.S_, axis=1).reshape(-1, 1))

        self.stability_indexes_ = Index

        self.A_ = (X.T).dot(np.linalg.pinv(self.S_))
        # self.A_ = np.dot(X.T , np.linalg.pinv(self.S_))

        #### 4. Plot the stability indexes of each final ICA components (optional)

        if plot:
            plt.figure(figsize=(10, 7))
            plt.plot(
                range(1, self.n_components + 1),
                self.stability_indexes_,
                linestyle="--",
                marker="o",
            )
            plt.title("Stability of ICA components")
            plt.xlabel("ICA components")
            plt.ylabel("Stability index")

        return

    def _parallel_decomposition(self, parallel, func, kwargs, algorithm, n_runs):
        """ Compute in parallel the n_runs runs of the ICA solver. If the solver comes from sklearn.FastICA, some potential convergence errors ar handled through 
        multiple retryings.
        
        Parameters
        ----------
        parallel : joblib.Parallel object 
            Object to use workers to compute in parallel the n_runs application of the function func to solve the ICA problem.
            
        func : callable
            Function to perform the ICA decomposition for a single run. It should return an array of ICA components of shape (n_components , n_observations)
            
        kwargs : dict
            A dictionnary of arguments to pass to the function func.
            
        algorithm : algorithm : str {'fastica_par' , 'fastica_def' , 'fastica_picard' , 'infomax' , 'infomax_ext' , 'infomax_orth'}
            See fit method.
            
        n_runs : int
            Number of times we run the FastICA algorithm (see fit method).

        Returns
        -------
        decomposition : list of arrays of shape (n_components , n_observations), length n_runs
            List of ICA sources obtained at each run.
        """

        if algorithm in ["fastica_par", "fastica_def"]:
            maxtrials = 10
            for i in range(maxtrials):
                try:
                    decomposition = parallel(
                        delayed(func)(**kwargs) for _ in range(n_runs)
                    )
                except ValueError:
                    if i < maxtrials - 1:
                        print(
                            "FastICA from sklearn did not converge due to numerical instabilities - Retrying..."
                        )
                        continue
                    else:
                        print("Too many attempts : FastICA did not converge !")
                        raise
                break
        else:
            decomposition = parallel(delayed(func)(**kwargs) for _ in range(n_runs))

        return decomposition

    def _ICA_decomposition(self, X_w):
        """ Apply FastICA or infomax (picard package) algorithm to the whitened matrix X_w to solve the ICA problem.
        
        Parameters
        ----------
        X_w : 2D array, shape (n_observations , n_components)
            Whitened data matrix.

        Returns
        -------
        S : 2D array, shape (n_components , n_observations)
            Array of sources obtained from a single run of an ICA solver. Each line corresponds to an ICA component.
        """

        if self._method == "picard":
            _, _, S = picard(
                X_w.T,
                max_iter=self.max_iter,
                whiten=False,
                centering=False,
                **self._solver_params
            )
        else:
            ica = FastICA(max_iter=self.max_iter, whiten=False, **self._solver_params)
            S = ica.fit_transform(X_w).T
        return S

    def _ICA_decomposition_bootstrap(self, X, whitening_params):
        """ Draw a bootstrap sample from the original data matrix X, whiten it and apply FastICA or infomax (picard package) algorithm 
        to solve the ICA problem.
        
        Parameters
        ----------
        X : 2D array, shape (n_observations , n_mixtures)
            Original data matrix.
            
        whitening_params : dict
            A dictionnary containing the arguments to pass to the whitening function to whiten the bootstrap matrix.

        Returns
        -------
        S : 2D array, shape (n_components , n_observations)
            Array of sources obtained from a single run of an ICA solver and a bootstrap sample of the original matrix X.
            Each line corresponds to an ICA component.
        """

        n_mixtures = X.shape[1]
        Xb = X[:, np.random.choice(range(n_mixtures), size=n_mixtures)]
        Xb_w = whitening(Xb, n_components=self.n_components, **whitening_params)

        if self._method == "picard":
            _, _, S = picard(
                Xb_w.T,
                max_iter=self.max_iter,
                whiten=False,
                centering=False,
                **self._solver_params
            )
        else:
            ica = FastICA(max_iter=self.max_iter, whiten=False, **self._solver_params)
            S = ica.fit_transform(Xb_w).T
        return S

    def _ICA_decomposition_fast_bootstrap(self, U, SVt):
        """ Draw a boostrap whitened sample from the original matrix X (svd decomposition of X = USVt) [1], and apply FastICA or infomax (picard package) algorithm 
        to solve the ICA problem.
        
        Parameters
        ----------
        U : 2D array, shape (n_observations , n_mixtures)
            
        SVt : 2D array, shape (n_mixtures , n_mixtures)

        Returns
        -------
        S : 2D array, shape (n_components , n_observations)
            Array of sources obtained from a single run of an ICA solver and a bootstrap sample of the original matrix X.
            Each line corresponds to an ICA component.
            
        References
        ----------
        [1] : Fisher A, Caffo B, Schwartz B, Zipunnikov V. Fast, Exact Bootstrap Principal Component Analysis for p > 1 million.
        J Am Stat Assoc. 2016;111(514):846-860. doi: 10.1080/01621459.2015.1062383. Epub 2016 Aug 18. PMID: 27616801; PMCID: PMC5014451.
        """

        n, p = SVt.shape[1], U.shape[0]
        Ab, Sb, Rbt = linalg.svd(SVt[:, np.random.choice(range(n), size=n)])
        Ub = np.dot(U, Ab)
        Ub, Rbt = svd_flip(Ub, Rbt)
        Xb_w = Ub[:, : self.n_components] * np.sqrt(p - 1)

        if self._method == "picard":
            _, _, S = picard(
                Xb_w.T,
                max_iter=self.max_iter,
                whiten=False,
                centering=False,
                **self._solver_params
            )
        else:
            ica = FastICA(max_iter=self.max_iter, whiten=False, **self._solver_params)
            S = ica.fit_transform(Xb_w).T
        return S

    def projection(self, method="mds", ax=None):
        """Plot the ``n_components*n_runs`` ICA components computed during fit() in 2D.
        Approximate the original dissimilarities between components by Euclidean distance.
        Each cluster is represented with a different color.
           
        Parameters
        ----------        
        method : string, optional
            Name of the dimensionality reduction method (e.g "tsne" , "mds" or "umap")
            The default is "umap".
            
        ax : matplotlib.axes, optional
            The default is None.
            
        Returns
        -------
        None.
        
        Notes
        -----
        - We use the dissimilarity measure ``sqrt(1 - |rho_ij|)`` (rho the Pearson correlation) instead of ``1 - |rho_ij|`` to reduce overlapping.
        
        - Please note that multidimensional scaling (MDS) is more computationally demanding than t-SNE or UMAP. However it takes into account the global structures of the data set while the others don't. For t-SNE or UMAP one cannot really interpret the inter-cluster distances.
        """

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        elif not isinstance(ax, matplotlib.axes.Axes):
            warnings.warn(
                "ax should be a matplotlib.axes.Axes object. It was redefined by default."
            )
            fig, ax = plt.subplots(figsize=(10, 6))

        if method == "tsne":
            embedding = manifold.TSNE(n_components=2, metric="precomputed")
        elif method == "mds":
            embedding = manifold.MDS(
                n_components=2, dissimilarity="precomputed", n_jobs=-1
            )
        elif method == "umap":
            embedding = umap.UMAP(n_components=2, metric="precomputed")

        P = embedding.fit_transform(np.sqrt(1 - self._Sim))

        ax.scatter(P[:, 0], P[:, 1], c=self._clusters, cmap="viridis")
        return


def MSTD(X, 
         m, 
         M, 
         step, 
         n_runs,
         fun="logcosh", 
         algorithm="fastica_par", 
         whiten=True, 
         max_iter=2000, 
         n_jobs=-1, 
         ax=None
):
    """Plot "MSTD graphs" to help choosing an optimal dimension for ICA decomposition.
        
    Run stabilized ICA algorithm for several dimensions in [m , M] and compute the
    stability distribution of the components each time.
       
    Parameters
    ----------
    X : 2D array, shape (n_observations , n_mixtures) 
        Training data
        
    m : int
        Minimal dimension for ICA decomposition.
        
    M : int > m
        Maximal dimension for ICA decomposition.
        
    step : int > 0
        Step between two dimensions (ex: if ``step = 2`` the function will test the dimensions
        m, m+2, m+4, ... , M).
        
    n_runs : int
        Number of times we run the FastICA algorithm (see fit method of class Stabilized_ICA)
    
    fun : str {'cube' , 'exp' , 'logcosh' , 'tanh'} or function, optional.
        The default is 'logcosh'. See the fit method of StabilizedICA for more details.
        
    algorithm : str {'fastica_par' , 'fastica_def' , 'fastica_picard' , 'infomax' , 'infomax_ext' , 'infomax_orth'}, optional.
        The algorithm applied for solving the ICA problem at each run. Please the supplementary explanations for more details.
        The default is 'fastica_par', i.e FastICA from sklearn with parallel implementation.
        
    whiten : bool, optional
        It True, X is whitened only once as an initial step, with a SVD solver and M components. If False, X must be already
        whitened, with M components. The default is True.
              
    max_iter : int, optional
        Parameter for _ICA_decomposition. The default is 2000.
    
    n_jobs : int
        Number of jobs to run in parallel for each stabilized ICA step. Default is -1
    
    ax : array of matplotlib.axes objects, optional
        The default is None.
            
    Returns
    -------
    None.
    
    References
    ----------
    Kairov U, Cantini L, Greco A, Molkenov A, Czerwinska U, Barillot E, Zinovyev A. Determining the optimal number of independent components for reproducible transcriptomic data analysis.
    BMC Genomics. 2017 Sep 11;18(1):712. doi: 10.1186/s12864-017-4112-9. PMID: 28893186; PMCID: PMC5594474.
    (see https://bmcgenomics.biomedcentral.com/track/pdf/10.1186/s12864-017-4112-9 ).
    
    Examples
    --------
    >>> from sica.base import MSTD
    >>> df = pd.read_csv("data.csv" , index_col = 0).transpose()  
    >>> MSTD(df.values , m = 5 , M = 100 , step = 2 , n_runs = 20 , max_iter = 2000)
    """
    if ax is None:
        fig, ax = plt.subplots(1, 2, figsize=(20, 7))
    else:
        try:
            ax = ax.flatten()
        except AttributeError:
            warnings.warn(
                "ax should be a numpy array containing at least two matplotlib.axes.Axes objects. It was redefined by default."
            )
            fig, ax = plt.subplots(1, 2, figsize=(20, 7))
        else:
            if len(ax) < 2:
                warnings.warn(
                    "ax is not of the right shape. It should contain at least two matplotlib.axes.Axes objects. It was redefined by default."
                )
                fig, ax = plt.subplots(1, 2, figsize=(20, 7))

    mean = []

    if whiten:
        X_w = whitening(
            X,
            n_components=M,
            svd_solver="full",
            chunked=False,
            chunk_size=None,
            zero_center=True,
        )
    else:
        X_w = as_float_array(X, copy=False)

    # for i in range(m , M+step , step): #uncomment if you don't want to use tqdm (and comment the line below !)
    for i in tqdm(range(m, M + step, step)):
        s = StabilizedICA(n_components = i, max_iter = max_iter, n_jobs = n_jobs)
        s.fit(X_w[:, :i], n_runs, fun = fun, algorithm = algorithm, whiten=False)
        mean.append(np.mean(s.stability_indexes_))
        ax[0].plot(range(1, len(s.stability_indexes_) + 1), s.stability_indexes_, "k")

    ax[1].plot(range(m, M + step, step), mean)

    ax[1].set_title("Mean stability")
    ax[1].set_xlabel("Number of components")
    ax[0].set_title("Index stability distribution")
    ax[0].set_xlabel("Number of components")
    return