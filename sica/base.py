import warnings
from typing import NoReturn, Optional, Tuple, Callable, List, Union, Any

import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import umap
from joblib import Parallel, delayed
from picard import picard
from scipy import linalg
from scipy.sparse import issparse
from sklearn import manifold
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import FastICA
from sklearn.utils import as_float_array, check_array
from sklearn.utils.extmath import svd_flip
from sklearn.utils.validation import FLOAT_DTYPES, check_is_fitted
from tqdm.notebook import tqdm

from ._whitening import whitening


def _check_algorithm(algorithm: str, fun: str) -> Tuple[str, dict]:
    all_algorithms = [
        "fastica_par",
        "fastica_def",
        "picard_fastica",
        "picard",
        "picard_ext",
        "picard_orth",
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
        "picard_fastica",
        "picard",
        "picard_ext",
        "picard_orth",
    ]:
        warnings.warn(
            "'logcosh' is not available for picard. By default, we assumed 'tanh' was the desired function"
        )
        fun = "tanh"

    if fun != "tanh" and algorithm in ["picard_fastica", "picard_ext"]:
        warnings.warn(
            "Using a different density than `'tanh'` may lead to erratic behavior of the picard algorithm"
            " when extended=True (see picard package for more explanations)"
        )

    if fun == "exp" and algorithm == "picard":
        warnings.warn(
            "Using the exponential density model may lead to a FloatingPointError. To solve this problem you may try "
            "to scale the non-linearity changing the alpha parameter in the exp density "
            " (ex : set the `fun` parameter of fit method to `picard.densities.Exp(params={'alpha': 0.1})`)"
        )

    if algorithm == "fastica_par":
        return "fastica", {"algorithm": "parallel", "fun": fun}
    elif algorithm == "fastica_def":
        return "fastica", {"algorithm": "deflation", "fun": fun}
    elif algorithm == "picard_fastica":
        return "picard", {"ortho": True, "extended": True, "fun": fun}
    elif algorithm == "picard":
        return "picard", {"ortho": False, "extended": False, "fun": fun}
    elif algorithm == "picard_ext":
        return "picard", {"ortho": False, "extended": True, "fun": fun}
    elif algorithm == "picard_orth":
        return "picard", {"ortho": True, "extended": False, "fun": fun}


def _centrotype(X: np.ndarray, Sim: np.ndarray, cluster_labels: list) -> np.ndarray:
    """Compute the centrotype of the cluster of ICA components defined by cluster_labels
    
       centrotype : component of the cluster which is the most similar to the other components
                   of the cluster
    Parameters
    ----------
    X : 2D array, shape (n_components , n_observations)
        matrix of independent ICA components
        
    Sim : 2D array, shape (n_components , n_components)
        similarity matrix for ICA components (i.e. rows of X)
        
    cluster_labels : list of integers
        indexes of the cluster of components (ex:[0 , 1 , 7] refers to the rows 0, 1 and 7 of X)

    Returns
    -------
    1D array, shape (n_observations)
        centrotype of the cluster of ICA components defined by cluster_labels

    """
    temp = np.argmax(np.sum(Sim[np.ix_(cluster_labels, cluster_labels)], axis=0))
    return X[cluster_labels[temp], :]


def _stability_index(Sim: np.ndarray, cluster_labels: list) -> float:
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


class StabilizedICA(BaseEstimator, TransformerMixin):
    """ Implement a stabilized version of the Independent Component Analysis algorithm.

    It fits the matrix factorization model X = AS, where A is the unmixing matrix (n_mixtures, n_sources), S is the
    source matrix (n_sources, n_observations) and X is the observed mixed data (n_mixtures, n_observations).

    Parameters
    ----------
    n_components : int
        Number of ICA components/sources.

    n_runs : int
            Number of times we run the FastICA algorithm

    resampling : str {None , 'bootstrap' , 'fast_bootstrap'}, optional
        Method for resampling the data before each run of the ICA solver.

        - If None, no resampling is applied.
        - If 'bootstrap' the classical bootstrap method is applied to the original data matrix, the resampled matrix is whitened (using the whitening hyperparameters set for the fit method) and the ICA components are extracted.
        - If 'fast_boostrap' a fast bootstrap algorithm is applied to the original data matrix and the whitening operation is performed simultaneously with SVD decomposition and then the ICA components are extracted (see References).

        Resampling could lead to quite heavy computations (whitening at each iteration), depending on the size of the input data. It should be considered with care. The default is None.

    algorithm : str {'fastica_par' , 'fastica_def' , 'picard_fastica' , 'picard' , 'picard_ext' , 'picard_orth'}, optional.
            The algorithm applied for solving the ICA problem at each run. Please see the supplementary explanations
            for more details. The default is 'fastica_par', i.e. FastICA from sklearn with parallel implementation.

    fun : str {'cube' , 'exp' , 'logcosh' , 'tanh'} or function, optional.

        If ``algorithm`` is in {'fastica_par' , 'fastica_def'}, it represents the functional form of the G function
        used in the approximation to neg-entropy. Could be either ‘logcosh’, ‘exp’, or ‘cube’.

        If ``algorithm`` is in {'picard_fastica' , 'picard' , 'picard_ext' , 'picard_orth'}, it is associated with
        the choice of a density model for the sources. See supplementary explanations for more details.

        The default is 'logcosh'.

    whiten : boolean, optional

        If True the matrix X is whitened, i.e. centered then projected in the space defined by its
        first ``n_components`` PCA components and reduced to unit variance along each of these axes.

        If False the input X matrix must be already whitened (the rows must be centered, scaled to unit
        variance and uncorrelated.)

        The default is True.

    max_iter : int
        Maximum number of iteration for the FastICA algorithm.

    plot : boolean, optional
        If True plot the stability indexes for each cluster in decreasing order.
        The default is False.

    normalize : boolean, optional
        If True normalize the rows of ``S_`` (i.e. the stabilized ICA components) to unit standard deviation.
        The default is True.

    reorientation : boolean,optional
        If True re-oriente the rows of ``S_`` towards positive heavy tail.
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

    n_jobs : int, optional
        Number of jobs to run in parallel. -1 means using all processors.
        See the joblib package documentation for more explanations. The default is 1.
    
    verbose: int, optional
        Control the verbosity: the higher, the more messages. The default is 0.
    
    Attributes
    ----------
    S_: 2D array, shape (n_components , n_observations)
        Array of sources/metagenes, each line corresponds to a stabilized ICA component (i.e. the centrotype of
        a cluster of components).  

    stability_indexes_ : 1D array, shape (n_components)
        Stability indexes for the stabilized ICA components.

    mean_ : 1D array, shape (n_mixtures)

    Notes
    -----
    Here `n_components` corresponds to the number of ICA sources, `n_mixtures` corresponds to the number
    of linear mixtures (i.e. linear mixtures of ICA sources) that we observe, and `n_observations` corresponds to the
    number of observations collected for these mixtures. Each time, the user needs to carefully determine which
    dimension in his data set should correspond to the linear mixtures of ICA sources and which dimension should
    correspond to the observations. The user should keep in mind that, at the end, he will obtain `n_components`
    vectors of dimension `n_observations`, independent form each other (as finite samples of latent independent
    distributions). The user guide and the definition of the ICA framework should be helpful.
    
    - For a data set of discretized sound signals registered by 10 microphones at 100 time points, if we want to retrieve 5 ICA sources we need to set `n_mixtures = 10`, `n_observations = 100` and `n_components = 5`.
    - For a gene expression data set with 100 samples and 10000 genes, if we want to retrieve 10 ICA sources **in the gene space** we need to set `n_mixtures = 100`, `n_observations = 10000` and `n_components = 10`.
    
    References
    ----------
    Fast bootstrap algorithm:
        Fisher A, Caffo B, Schwartz B, Zipunnikov V. Fast, Exact Bootstrap Principal Component Analysis for p > 1 million.
        J Am Stat Assoc. 2016;111(514):846-860. doi: 10.1080/01621459.2015.1062383. Epub 2016 Aug 18. PMID: 27616801; PMCID: PMC5014451.
        
    ICASSO method :
        J. Himberg and A. Hyvarinen, "Icasso: software for investigating the reliability of ICA estimates by clustering and visualization," 
        2003 IEEE XIII Workshop on Neural Networks for Signal Processing (IEEE Cat. No.03TH8718), 2003, pp. 259-268, doi: 10.1109/NNSP.2003.1318025
        (see https://www.cs.helsinki.fi/u/ahyvarin/papers/Himberg03.pdf). 

    Picard algorithm and extensions:
        Pierre Ablin, Jean-Francois Cardoso, Alexandre Gramfort, "Faster independent component analysis by
        preconditioning with Hessian approximations" IEEE Transactions on Signal Processing, 2018
        (see https://arxiv.org/abs/1706.08171).

        Pierre Ablin, Jean-François Cardoso, Alexandre Gramfort "Faster ICA under orthogonal constraint" ICASSP, 2018
        (see https://arxiv.org/abs/1711.10873)

    UMAP:
        For more details about the UMAP (Uniform Manifold Approximation and Projection), see https://pypi.org/project/umap-learn/.
    
    Examples
    --------
    >>> import pandas as pd
    >>> from sica.base import StabilizedICA   
    >>> df = pd.read_csv("data.csv" , index_col = 0)
    >>> sICA = StabilizedICA(n_components = 45 , n_runs = 30, plot = True, n_jobs = -1)
    >>> sICA.fit(df)
    >>> Sources = pd.DataFrame(sICA.S_ , columns = df.index , index = ['source ' + str(i) for i in range(sICA.S_.shape[0])])
    >>> Sources.head()                
    """

    def __init__(
            self,
            n_components: int,
            n_runs: int,
            resampling: Optional[Union[str, None]] = None,
            algorithm: Optional[str] = "fastica_par",
            fun: Optional[str] = "logcosh",
            whiten: Optional[bool] = True,
            max_iter: Optional[int] = 2000,
            plot: Optional[bool] = False,
            normalize: Optional[bool] = True,
            reorientation: Optional[bool] = True,
            pca_solver: Optional[str] = "auto",
            chunked: Optional[bool] = False,
            chunk_size: Optional[Union[int, None]] = None,
            zero_center: Optional[bool] = True,
            n_jobs: Optional[int] = 1,
            verbose: Optional[int] = 0,
    ) -> NoReturn:
        super().__init__()
        self.n_components = n_components
        self.n_runs = n_runs
        self.algorithm = algorithm
        self.fun = fun
        self.whiten = whiten
        self.plot = plot
        self.normalize = normalize
        self.reorientation = reorientation
        self.pca_solver = pca_solver
        self.chunked = chunked
        self.chunk_size = chunk_size
        self.zero_center = zero_center
        self.max_iter = max_iter
        self.resampling = resampling
        self.n_jobs = n_jobs
        self.verbose = verbose

        self.S_ = None
        self.mean_ = None
        self.stability_indexes_ = None

    def fit(self, X: np.ndarray, y: Optional[Any] = None) -> object:
        """ Fit the ICA model with X (use stabilization).
        
        1. Compute the ICA components of X ``n_runs`` times.
        
        2. Cluster all the ``n_components*n_runs`` components with agglomerative 
           hierarchical clustering (average linkage) into ``n_components`` clusters.
           
        3. For each cluster compute its stability index and return its centrotype as the
           final ICA component.              
                 
        Parameters
        ----------
        X : 2D array-like, shape (n_mixtures, n_observations) or (n_components, n_observations) if whiten is False.
            Training data 

        y : Ignored
            Ignored.

        Returns
        -------        
        self : object
            Returns the instance itself.
        """
        #### 0. Initialisation

        # Here we consider the transpose of X so that the rest of the code is in line with sklearn.decomposition.FastICA
        # which considers observations (or components for pre-whitened matrices) in rows and mixtures in columns.
        X = check_array(X, dtype=FLOAT_DTYPES, accept_sparse=True, copy=self.whiten).T

        n_observations, n_mixtures = X.shape
        Centrotypes = np.zeros((self.n_components, n_observations))
        Index = np.zeros(self.n_components)

        self._method, self._solver_params = _check_algorithm(self.algorithm, self.fun)

        parallel = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)

        #### 1. Compute the n_components*n_runs ICA components depending on the resampling strategy

        if self.resampling is None:

            # Pre-processing (whitening)
            if self.whiten:
                X_w, self.mean_ = whitening(
                    X,
                    n_components=self.n_components,
                    svd_solver=self.pca_solver,
                    chunked=self.chunked,
                    chunk_size=self.chunk_size,
                    zero_center=self.zero_center,
                )
            else:
                X_w = as_float_array(X, copy=False)
                self.mean_ = None

            # Compute the n_components*n_runs ICA components
            decomposition = self._parallel_decomposition(
                parallel=parallel,
                func=self._ICA_decomposition,
                kwargs={
                    "X_w": X_w,
                    "method": self._method,
                    "max_iter": self.max_iter,
                    "solver_params": self._solver_params
                },
            )

        elif self.resampling == "bootstrap":

            if not self.whiten:
                raise ValueError(
                    "The matrix X should not be pre-whitened when resampling = 'bootstrap'. The whitening step is "
                    "performed consecutively to each resampling (using the whitening hyperparameters set by the user). "
                )

            # Pre-processing (save whitening parameters)
            whitening_params = {
                "svd_solver": self.pca_solver,
                "chunked": self.chunked,
                "chunk_size": self.chunk_size,
                "zero_center": self.zero_center,
            }

            self.mean_ = np.mean(X, axis=0)

            # Compute the n_components*n_runs ICA components
            decomposition = self._parallel_decomposition(
                parallel=parallel,
                func=self._ICA_decomposition_bootstrap,
                kwargs={
                    "X": X,
                    "whitening_params": whitening_params,
                    "method": self._method,
                    "max_iter": self.max_iter,
                    "solver_params": self._solver_params,
                    "n_components": self.n_components
                },
            )

        elif self.resampling == "fast_bootstrap":

            if not self.whiten:
                raise ValueError(
                    "The matrix X should not be pre-whitened when resampling = 'fast_bootstrap'. The whitening step "
                    "is performed consecutively to each resampling (with SVD decomposition)."
                )

            elif issparse(X):
                raise ValueError(
                    "The 'fast_bootstrap' resampling algorithm cannot handle sparse matrices. Please provide a numpy."
                    "ndarray instead."
                )

            # Pre-processing (initial svd decomposition)
            self.mean_ = np.mean(X, axis=0)
            U, S, Vt = linalg.svd(X - self.mean_, full_matrices=False)
            SVt = np.dot(np.diag(S), Vt)

            # Compute the n_components*n_runs ICA components
            decomposition = self._parallel_decomposition(
                parallel=parallel,
                func=self._ICA_decomposition_fast_bootstrap,
                kwargs={
                    "U": U,
                    "SVt": SVt,
                    "method": self._method,
                    "max_iter": self.max_iter,
                    "solver_params": self._solver_params,
                    "n_components": self.n_components
                },
            )

        else:
            raise ValueError(
                "Unrecognized resampling method. Please choose among None, 'bootstrap' or 'fast_bootstrap'"
            )

        self._components = np.vstack(decomposition)

        #### 2. Cluster the n_components*n_runs ICA components with hierarchical clustering

        # Compute Similarity matrix between ICA components (Pearson correlation)
        self._Sim = np.abs(np.corrcoef(x=self._components, rowvar=True))

        # Cluster the components with hierarchical clustering
        clustering = AgglomerativeClustering(
            n_clusters=self.n_components, affinity="precomputed", linkage="average"
        ).fit(1 - self._Sim)
        self._clusters = clustering.labels_

        #### 3. For each cluster compute the stability index and the centrotype

        for i in range(self.n_components):
            cluster_labels = list(np.argwhere(clustering.labels_ == i).flatten())
            Centrotypes[i, :] = _centrotype(self._components, self._Sim, cluster_labels)
            Index[i] = _stability_index(self._Sim, cluster_labels)

        # Sort the centrotypes (i.e. final components) by stability index
        indices = np.argsort(-1 * Index)
        Centrotypes, Index = Centrotypes[indices, :], Index[indices]

        # Re-oriente the stabilized ICA components towards positive heaviest tails
        if self.reorientation:
            self.S_ = (
                          np.where(stats.skew(Centrotypes, axis=1) >= 0, 1, -1).reshape(-1, 1)
                      ) * Centrotypes
        else:
            self.S_ = Centrotypes

        # Normalize the stabilized ICA components to unit variance
        if self.normalize:
            self.S_ = self.S_ / (np.std(self.S_, axis=1).reshape(-1, 1))

        self.stability_indexes_ = Index

        #### 4. Plot the stability indexes of each final ICA components (optional)

        if self.plot:
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

        return self

    def _parallel_decomposition(self,
                                parallel: Parallel,
                                func: Callable[..., np.ndarray],
                                kwargs: dict
                                ) -> List[np.ndarray]:
        """ Compute in parallel the n_runs runs of the ICA solver. If the solver comes from sklearn.FastICA,
        some potential convergence errors ar handled through multiple retryings.
        
        Parameters
        ----------
        parallel : joblib.Parallel
            Object to use workers to compute in parallel the n_runs application of the function func to solve the ICA
            problem.
            
        func : callable
            Function to perform the ICA decomposition for a single run. It should return an array of ICA components of
            shape (n_components , n_observations)

        kwargs : dict
            A dictionnary of arguments to pass to the function func.

        Returns
        -------
        decomposition : list of arrays of shape (n_components , n_observations), length n_runs
            List of ICA sources obtained at each run.
        """

        if self.algorithm in ["fastica_par", "fastica_def"]:
            maxtrials = 10
            attempt = 1
            success = False
            decomposition = None
            while (attempt <= maxtrials) and (not success):
                try:
                    decomposition = parallel(delayed(func)(**kwargs) for _ in range(self.n_runs))
                    success = True
                except ValueError:
                    print("FastICA from sklearn did not converge due to numerical instabilities - Retrying...")
                attempt += 1
            if not success:
                raise ValueError("Too many attempts: FastICA did not converge !")

        else:
            decomposition = parallel(delayed(func)(**kwargs) for _ in range(self.n_runs))

        return decomposition

    @staticmethod
    def _ICA_decomposition(X_w: np.ndarray, method: str, max_iter: int, solver_params: dict) -> np.ndarray:
        """ Apply FastICA or picard (picard package) algorithm to the whitened matrix X_w to solve the ICA problem.
        
        Parameters
        ----------
        X_w : 2D array, shape (n_observations , n_components)
            Whitened data matrix.

        Returns
        -------
        S : 2D array, shape (n_components , n_observations)
            Array of sources obtained from a single run of an ICA solver. Each line corresponds to an ICA component.
        """

        if method == "picard":
            _, _, S = picard(
                X_w.T,
                max_iter=max_iter,
                whiten=False,
                centering=False,
                **solver_params
            )
        else:
            ica = FastICA(max_iter=max_iter, whiten=False, **solver_params)
            S = ica.fit_transform(X_w).T
        return S

    @staticmethod
    def _ICA_decomposition_bootstrap(
            X: np.ndarray,
            whitening_params: dict,
            method: str,
            max_iter: int,
            solver_params: dict,
            n_components: int) -> np.ndarray:
        """ Draw a bootstrap sample from the original data matrix X, whiten it and apply FastICA or picard
        (picard package) algorithm to solve the ICA problem.
        
        Parameters
        ----------
        X : 2D array, shape (n_observations , n_mixtures)
            Original data matrix.
            
        whitening_params : dict
            A dictionnary containing the arguments to pass to the whitening function to whiten the bootstrap matrix.

        Returns
        -------
        S : 2D array, shape (n_components , n_observations)
            Array of sources obtained from a single run of an ICA solver and a bootstrap sample of the original matrix
            X. Each line corresponds to an ICA component.
        """

        n_mixtures = X.shape[1]
        Xb = X[:, np.random.choice(range(n_mixtures), size=n_mixtures)]
        Xb_w, _ = whitening(Xb, n_components=n_components, **whitening_params)

        if method == "picard":
            _, _, S = picard(
                Xb_w.T,
                max_iter=max_iter,
                whiten=False,
                centering=False,
                **solver_params
            )
        else:
            ica = FastICA(max_iter=max_iter, whiten=False, **solver_params)
            S = ica.fit_transform(Xb_w).T
        return S

    @staticmethod
    def _ICA_decomposition_fast_bootstrap(
            U: np.ndarray,
            SVt: np.ndarray,
            method: str,
            max_iter: int,
            solver_params: dict,
            n_components: int) -> np.ndarray:
        """ Draw a boostrap whitened sample from the original matrix X (svd decomposition of X = USVt) [1], and apply
        FastICA or picard (picard package) algorithm to solve the ICA problem.
        
        Parameters
        ----------
        U : 2D array, shape (n_observations , n_mixtures)
            
        SVt : 2D array, shape (n_mixtures , n_mixtures)

        Returns
        -------
        S : 2D array, shape (n_components , n_observations)
            Array of sources obtained from a single run of an ICA solver and a bootstrap sample of the original matrix
            X. Each line corresponds to an ICA component.
            
        References
        ----------
        [1] : Fisher A, Caffo B, Schwartz B, Zipunnikov V. Fast, Exact Bootstrap Principal Component Analysis for p > 1 million.
        J Am Stat Assoc. 2016;111(514):846-860. doi: 10.1080/01621459.2015.1062383. Epub 2016 Aug 18. PMID: 27616801; PMCID: PMC5014451.
        """

        n, p = SVt.shape[1], U.shape[0]
        Ab, Sb, Rbt = linalg.svd(SVt[:, np.random.choice(range(n), size=n)])
        Ub = np.dot(U, Ab)
        Ub, Rbt = svd_flip(Ub, Rbt)
        Xb_w = Ub[:, : n_components] * np.sqrt(p - 1)

        if method == "picard":
            _, _, S = picard(
                Xb_w.T,
                max_iter=max_iter,
                whiten=False,
                centering=False,
                **solver_params
            )
        else:
            ica = FastICA(max_iter=max_iter, whiten=False, **solver_params)
            S = ica.fit_transform(Xb_w).T
        return S

    def transform(self, X: np.ndarray, copy: Optional[bool] = True) -> np.ndarray:
        """ Apply dimensionality reduction to X (i.e. recover the mixing matrix applying the pseudo-inverse
        of the sources).

        Parameters
        ----------
        X : 2D array-like, shape (n_mixtures, n_observations)

        copy: bool, optional
            If False, data passed to fit are overwritten. The default is True.

        Returns
        -------
        A : 2D array, shape (n_mixtures, n_components)
            Unmixing matrix which maps the independent sources to the data (i.e. X.T = AS)

        """
        X = check_array(X, dtype=FLOAT_DTYPES, accept_sparse=True, copy=copy).T
        check_is_fitted(self)

        # Warning: problem which needs to be addressed !!
        # if self.mean_ is not None:
        #    X -= self.mean_

        A = X.T.dot(np.linalg.pinv(self.S_))
        return A

    def inverse_transform(self, X: np.ndarray, copy: Optional[bool] = True) -> np.ndarray:
        """ Transform the mixing matrix back to the mixed data (applying the sources).

        Parameters
        ----------
        X: 2D array-like, shape (n_mixtures , n_components)

        copy: bool, optional
            If False, data passed to fit are overwritten. The default is True.

        Returns
        -------
        X_new: 2D array, shape (n_mixtures, n_observations)
        """
        check_is_fitted(self)
        X_new = np.dot(X, self.S_)

        # Warning: problem which needs to be addressed !!
        # if self.mean_ is not None:
        #    X_new += self.mean_.reshape(1, -1)

        return X_new

    def projection(
            self,
            method: Optional[str] = "mds",
            ax: Optional[Union[matplotlib.axes.Axes, None]] = None
    ) -> None:
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
        
        - Please note that multidimensional scaling (MDS) is more computationally demanding than t-SNE or UMAP. However, it takes into account the global structures of the data set while the others don't. For t-SNE or UMAP one cannot really interpret the inter-cluster distances.
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
        else:
            raise ValueError("method parameter value can only be 'tsne', 'mds' or 'umap'")

        P = embedding.fit_transform(np.sqrt(1 - self._Sim))

        ax.scatter(P[:, 0], P[:, 1], c=self._clusters, cmap="viridis")
        return


def MSTD(X: np.ndarray,
         m: int,
         M: int,
         step: int,
         n_runs: int,
         fun: Optional[str] = "logcosh",
         algorithm: Optional[str] = "fastica_par",
         whiten: Optional[bool] = True,
         max_iter: Optional[int] = 2000,
         n_jobs: Optional[int] = -1,
         ax: Optional[Union[matplotlib.axes.Axes, None]] = None
         ) -> None:
    """Plot "MSTD graphs" to help choose an optimal dimension for ICA decomposition.
        
    Run stabilized ICA algorithm for several dimensions in [m , M] and compute the
    stability distribution of the components each time.
       
    Parameters
    ----------
    X : 2D array, shape (n_mixtures, n_observations)
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
        
    algorithm : str {'fastica_par' , 'fastica_def' , 'picard_fastica' , 'picard' , 'picard_ext' , 'picard_orth'}, optional.
        The algorithm applied for solving the ICA problem at each run. Please the supplementary explanations for more
        details. The default is 'fastica_par', i.e. FastICA from sklearn with parallel implementation.
        
    whiten : bool, optional
        It True, X is whitened only once as an initial step, with an SVD solver and M components. If False, X must be
        already whitened, with M components. The default is True.
              
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
    >>> import pandas as pd
    >>> from sica.base import MSTD
    >>> df = pd.read_csv("data.csv" , index_col = 0)
    >>> MSTD(df.values , m = 5 , M = 100 , step = 2 , n_runs = 20)
    """
    if ax is None:
        fig, ax = plt.subplots(1, 2, figsize=(20, 7))
    else:
        try:
            ax = ax.flatten()
        except AttributeError:
            warnings.warn(
                "ax should be a numpy array containing at least two matplotlib.axes.Axes objects. It was redefined by "
                "default. "
            )
            fig, ax = plt.subplots(1, 2, figsize=(20, 7))
        else:
            if len(ax) < 2:
                warnings.warn(
                    "ax is not of the right shape. It should contain at least two matplotlib.axes.Axes objects. It "
                    "was redefined by default. "
                )
                fig, ax = plt.subplots(1, 2, figsize=(20, 7))

    # Here we consider the transpose of X so that the rest of the code is in line with sklearn.decomposition.FastICA
    # which considers observations (or components for pre-whitened matrices) in rows and mixtures in columns.
    X = check_array(X, dtype=FLOAT_DTYPES, accept_sparse=True).T
    mean = []

    if whiten:
        X_w, _ = whitening(
            X,
            n_components=M,
            svd_solver="auto",
            chunked=False,
            chunk_size=None,
            zero_center=True,
        )
    else:
        X_w = as_float_array(X, copy=False)

    # for i in range(m , M+step , step): #uncomment if you don't want to use tqdm (and comment the line below !)
    for i in tqdm(range(m, M + step, step)):
        s = StabilizedICA(n_components=i, n_runs=n_runs, algorithm=algorithm, fun=fun, whiten=False,
                          max_iter=max_iter, n_jobs=n_jobs)
        s.fit(X_w[:, :i].T)
        mean.append(np.mean(s.stability_indexes_))
        ax[0].plot(range(1, len(s.stability_indexes_) + 1), s.stability_indexes_, "k")

    ax[1].plot(range(m, M + step, step), mean)

    ax[1].set_title("Mean stability")
    ax[1].set_xlabel("Number of components")
    ax[0].set_title("Index stability distribution")
    ax[0].set_xlabel("Number of components")
    return
