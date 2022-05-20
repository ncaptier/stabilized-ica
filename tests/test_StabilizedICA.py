import itertools
import pytest

from numpy.testing import assert_almost_equal

import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
from sica.base import StabilizedICA

from typing import Optional, Union, Tuple


def center_and_norm(x: np.ndarray, axis: Optional[int] = -1) -> None:
    """Centers and norms x **in place**
    Parameters
    -----------
    x: ndarray
        Array with an axis of observations (statistical units) measured on
        random variables.
    axis: int, optional
        Axis along which the mean and variance are calculated.
    """
    x = np.rollaxis(x, axis)
    x -= x.mean(axis=0)
    x /= x.std(axis=0)
    return


def create_data_superGauss(
        add_noise: bool,
        seed: int,
        N: Optional[int] = 3,
        T: Optional[int] = 1000,
        M: Optional[Union[int, None]] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.random.RandomState]:
    """ Create a simple data set with superGaussian-distributed sources (Laplace distribution).  
    Parameters
    ----------
    add_noise : bool
    seed : int
        Initialization of the pseudo-random number generator (np.random.RandomState). Used for generating the sources.
    N : int, optional
        Number of sources. The default is 3.
    T : int, optional
        Number of observations. The default is 1000.
    M : {int , None} , optional
        Number of mixtures. If None we set M = N. The default is None.
    Returns
    -------
    X : array, shape (M , T)
        Observed data set.
    A : array, shape (M , N)
        Mixture array.
    S : array, shape (N , T)
        Sources array.
    rng : np.random.RandomState
    """
    rng = np.random.RandomState(seed)
    S = rng.laplace(size=(N, T))
    center_and_norm(S)
    if M is None:
        A = rng.randn(N, N)
    else:
        A = rng.randn(M, N)
    X = np.dot(A, S)

    if add_noise:
        if M is None:
            X += 0.1*rng.randn(N, T)
        else:
            X += 0.1*rng.randn(M, T)

    return X, A, S, rng


def create_data_subGauss(
        add_noise: int,
        seed: int,
        N: Optional[int] = 3,
        T: Optional[int] = 1000,
        M: Optional[Union[int, None]] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.random.RandomState]:
    """ Create a simple data set with subGaussian-distributed sources (uniform distribution).
    """
    rng = np.random.RandomState(seed)
    S = rng.uniform(low=-1, high=1, size=(N, T))
    center_and_norm(S)
    if M is None:
        A = rng.randn(N, N)
    else:
        A = rng.randn(M, N)
    X = np.dot(A, S)

    if add_noise:
        if M is None:
            X += 0.1*rng.randn(N, T)
        else:
            X += 0.1*rng.randn(M, T)

    return X, A, S, rng


def create_data_mix(
        add_noise: int,
        seed: int,
        N: Optional[int] = 3,
        T: Optional[int] = 1000,
        M: Optional[Union[int, None]] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.random.RandomState]:
    """ Create a simple data set with a mixture of superGaussian-distributed and 
    subGaussian-distributed sources (Laplace and uniform distributions). 
    """
    rng = np.random.RandomState(seed)
    S = np.zeros((3, 1000))
    S[: int(N // 2), :] = rng.uniform(low=-1, high=1, size=(int(N // 2), T))
    S[int(N // 2):, :] = rng.laplace(size=(N - int(N // 2), T))
    center_and_norm(S)
    if M is None:
        A = rng.randn(N, N)
    else:
        A = rng.randn(M, N)
    X = np.dot(A, S)

    if add_noise:
        if M is None:
            X += 0.1*rng.randn(N, T)
        else:
            X += 0.1*rng.randn(M, T)

    return X, A, S, rng


def base_test_simple(
        strategy: Tuple[str, str],
        data: Tuple[np.ndarray, np.ndarray, np.ndarray, np.random.RandomState],
        noise: bool
) -> None:
    """ Simple test function for stabilized ICA without resampling.
    Parameters
    ----------
    strategy : (string , string)
        Pair of algorithm and function which characterizes the solver of the ICA problem
        (ex : ('fastica_par' , 'logcosh')).

    data : array X, array A, array S, randomstate 
        Simulated data.

    noise : bool
        Indicate whether the data have been simulated with noise.

    Returns
    -------
    None.

    """
    X, A, S, rng = data
    n_samples = S.shape[1]

    whitening = [True, False]
    normalizing = [True, False]
    reorienting = [True, False]
    for whiten, norm, reorient in itertools.product(
        whitening, normalizing, reorienting
    ):
        sica = StabilizedICA(n_components=S.shape[0],
                             n_runs=10,
                             fun=strategy[1],
                             algorithm=strategy[0],
                             whiten=whiten,
                             normalize=norm,
                             reorientation=reorient,
                             resampling=None,
                             n_jobs=1,
                             verbose=0
                             )
        if whiten:
            test_A = sica.fit_transform(X)
        else:
            pca = PCA(n_components=S.shape[0], whiten=True, random_state=rng)
            Xw = pca.fit_transform(X.T)
            test_Aw = sica.fit_transform(Xw.T)

        assert sica.S_.shape == S.shape
        assert sica.stability_indexes_.shape == (S.shape[0],)

        # Check that the mixing model described in the docstring holds:
        if not noise:
            if whiten:
                assert test_A.shape == A.shape
                assert_almost_equal(X, np.dot(test_A, sica.S_) + sica.mean_.reshape(-1, 1))
            else:
                assert test_Aw.shape == (S.shape[0], S.shape[0])
                assert_almost_equal(X, pca.inverse_transform(np.dot(test_Aw, sica.S_).T).T)

        center_and_norm(sica.S_)
        s1_, s2_, s3_ = sica.S_
        # Check to see if the sources have been estimated
        # in the wrong order
        ranks = np.argsort(abs(np.dot(s1_, S.T)))[::-1]
        if abs(np.dot(s2_, S[ranks[2], :])) > abs(np.dot(s2_, S[ranks[1], :])):
            temp = ranks[1]
            ranks[1] = ranks[2]
            ranks[2] = temp
        s1, s2, s3 = S[ranks, :]

        s1_ *= np.sign(np.dot(s1_, s1))
        s2_ *= np.sign(np.dot(s2_, s2))
        s3_ *= np.sign(np.dot(s3_, s3))

        # Check that we have estimated the original sources
        if noise:
            assert_almost_equal(np.dot(s1_, s1) / n_samples, 1, decimal=1)
            assert_almost_equal(np.dot(s2_, s2) / n_samples, 1, decimal=1)
            assert_almost_equal(np.dot(s3_, s3) / n_samples, 1, decimal=1)
        else:
            assert_almost_equal(np.dot(s1_, s1) / n_samples, 1, decimal=2)
            assert_almost_equal(np.dot(s2_, s2) / n_samples, 1, decimal=2)
            assert_almost_equal(np.dot(s3_, s3) / n_samples, 1, decimal=2)


def base_test_with_bootstrap(
        resampling: str,
        strategy: Tuple[str, str],
        data: Tuple[np.ndarray, np.ndarray, np.ndarray, np.random.RandomState],
        noise: bool
) -> None:
    """ Simple test function for  StabilizedICA with resampling.
    Parameters
    ----------
    resampling : {'bootstrap' , 'fast_bootstrap'}
        Resampling strategy.
    strategy : (string , string)
        Pair of algorithm and function which characterizes the solver of the ICA problem
        (ex : ('fastica_par' , 'logcosh')).
    data : array X, array A, array S, randomstate 
        Simulated data.

    noise : bool
        Indicate whether the data have been simulated with noise.

    Returns
    -------
    None.

    """
    X, A, S, rng = data
    n_samples = S.shape[1]

    sica = StabilizedICA(
        n_components=S.shape[0],
        n_runs=10,
        fun=strategy[1],
        algorithm=strategy[0],
        resampling=resampling,
        n_jobs=-1,
        verbose=0,
    )

    test_A = sica.fit_transform(X)

    assert test_A.shape == A.shape
    assert sica.S_.shape == S.shape
    assert sica.stability_indexes_.shape == (S.shape[0],)

    # Check that the mixing model described in the docstring holds:
    if not noise:
        assert_almost_equal(X, np.dot(test_A, sica.S_) + sica.mean_.reshape(-1, 1))

    center_and_norm(sica.S_)
    s1_, s2_, s3_ = sica.S_
    # Check to see if the sources have been estimated
    # in the wrong order
    ranks = np.argsort(abs(np.dot(s1_, S.T)))[::-1]
    if abs(np.dot(s2_, S[ranks[2], :])) > abs(np.dot(s2_, S[ranks[1], :])):
        temp = ranks[1]
        ranks[1] = ranks[2]
        ranks[2] = temp
    s1, s2, s3 = S[ranks, :]

    s1_ *= np.sign(np.dot(s1_, s1))
    s2_ *= np.sign(np.dot(s2_, s2))
    s3_ *= np.sign(np.dot(s3_, s3))

    # Check that we have estimated the original sources
    assert_almost_equal(np.dot(s1_, s1) / n_samples, 1, decimal=1)
    assert_almost_equal(np.dot(s2_, s2) / n_samples, 1, decimal=1)
    assert_almost_equal(np.dot(s3_, s3) / n_samples, 1, decimal=1)

    with pytest.raises(ValueError):
        sica_wf = StabilizedICA(
            n_components=S.shape[0],
            n_runs=10,
            fun=strategy[1],
            whiten=False,
            algorithm=strategy[0],
            resampling=resampling,
            n_jobs=-1,
            verbose=0,
        )
        sica_wf.fit(X)


@pytest.mark.parametrize("add_noise", [True, False])
@pytest.mark.parametrize("seed", range(2))
@pytest.mark.parametrize(
    "strategy",
    [
        ("fastica_par", "exp"),
        ("fastica_par", "logcosh"),
        ("fastica_def", "exp"),
        ("fastica_def", "logcosh"),
        ("infomax", "tanh"),
        ("infomax_orth", "exp"),
        ("infomax_orth", "tanh"),
        ("fastica_picard", "exp"),
        ("fastica_picard", "tanh"),
        ("infomax_ext", "exp"),
        ("infomax_ext", "tanh"),
    ],
)
def test_StabilizedICA_supGaussian(add_noise: bool, seed: bool, strategy: Tuple[str, str]):
    data = create_data_superGauss(add_noise, seed, M=100)
    base_test_simple(strategy=strategy, data=data, noise=add_noise)


@pytest.mark.parametrize("add_noise", [True, False])
@pytest.mark.parametrize("seed", range(2))
@pytest.mark.parametrize(
    "strategy",
    [
        ("fastica_par", "cube"),
        ("fastica_def", "cube"),
        ("infomax", "cube"),
        ("infomax_orth", "cube"),
        ("fastica_picard", "tanh"),
        ("infomax_ext", "tanh"),
    ],
)
def test_StabilizedICA_subGaussian(add_noise: bool, seed: bool, strategy: Tuple[str, str]):
    data = create_data_subGauss(add_noise, seed, M=10)
    base_test_simple(strategy=strategy, data=data, noise=add_noise)


@pytest.mark.parametrize("add_noise", [True, False])
@pytest.mark.parametrize("seed", range(2))
@pytest.mark.parametrize(
    "strategy",
    [
        ("fastica_par", "cube"),
        ("fastica_par", "exp"),
        ("fastica_par", "logcosh"),
        ("fastica_def", "cube"),
        ("fastica_def", "exp"),
        ("fastica_def", "logcosh"),
        ("fastica_picard", "tanh"),
        ("infomax_ext", "tanh"),
    ],
)
def test_StabilizedICA_mix(add_noise: bool, seed: bool, strategy: Tuple[str, str]):
    data = create_data_mix(add_noise, seed)
    base_test_simple(strategy=strategy, data=data, noise=add_noise)


@pytest.mark.parametrize("add_noise", [True, False])
@pytest.mark.parametrize("seed", range(2))
@pytest.mark.parametrize(
    "strategy",
    [
        ("fastica_par", "exp"),
        ("fastica_par", "logcosh"),
        ("fastica_def", "exp"),
        ("fastica_def", "logcosh"),
        ("infomax", "tanh"),
        ("infomax_orth", "exp"),
        ("infomax_orth", "tanh"),
        ("fastica_picard", "exp"),
        ("fastica_picard", "tanh"),
        ("infomax_ext", "exp"),
        ("infomax_ext", "tanh"),
    ],
)
@pytest.mark.parametrize("resampling", ["bootstrap", "fast_bootstrap"])
@pytest.mark.slow
def test_StabilizedICA_bootstrap_supGaussian(add_noise: bool, seed: bool, strategy: Tuple[str, str], resampling: str):
    data = create_data_superGauss(add_noise, seed, M=100)
    base_test_with_bootstrap(resampling=resampling, strategy=strategy, data=data, noise=add_noise)


@pytest.mark.parametrize("add_noise", [True, False])
@pytest.mark.parametrize("seed", range(2))
@pytest.mark.parametrize(
    "strategy",
    [
        ("fastica_par", "cube"),
        ("fastica_def", "cube"),
        ("infomax", "cube"),
        ("infomax_orth", "cube"),
        ("fastica_picard", "tanh"),
        ("infomax_ext", "tanh"),
    ],
)
@pytest.mark.parametrize("resampling", ["bootstrap", "fast_bootstrap"])
@pytest.mark.slow
def test_StabilizedICA_bootstrap_subGaussian(add_noise: bool, seed: bool, strategy: Tuple[str, str], resampling: str):
    data = create_data_subGauss(add_noise, seed, M=100)
    base_test_with_bootstrap(resampling=resampling, strategy=strategy, data=data, noise=add_noise)


@pytest.mark.parametrize("add_noise", [True, False])
@pytest.mark.parametrize("seed", range(2))
@pytest.mark.parametrize(
    "strategy",
    [
        ("fastica_par", "cube"),
        ("fastica_par", "exp"),
        ("fastica_par", "logcosh"),
        ("fastica_def", "cube"),
        ("fastica_def", "exp"),
        ("fastica_def", "logcosh"),
        ("fastica_picard", "tanh"),
        ("infomax_ext", "tanh"),
    ],
)
@pytest.mark.parametrize("resampling", ["bootstrap", "fast_bootstrap"])
@pytest.mark.slow
def test_StabilizedICA_bootstrap_mix(add_noise: bool, seed: bool, strategy: Tuple[str, str], resampling: str):
    data = create_data_mix(add_noise, seed, M=100)
    base_test_with_bootstrap(resampling=resampling, strategy=strategy, data=data, noise=add_noise)


def test_reorientation():
    X, A, S, rng = create_data_superGauss(add_noise=False, seed=42)
    sica = StabilizedICA(
        n_components=3, n_runs=10, reorientation=True, resampling=None, n_jobs=1, verbose=0
    )
    sica.fit(X)

    assert stats.skew(sica.S_[0, :]) >= 0
    assert stats.skew(sica.S_[1, :]) >= 0
    assert stats.skew(sica.S_[2, :]) >= 0


def test_normalization():
    X, A, S, rng = create_data_superGauss(add_noise=False, seed=42)
    sica = StabilizedICA(
        n_components=3, n_runs=10, normalize=True, resampling=None, n_jobs=1, verbose=0
    )
    sica.fit(X)

    assert_almost_equal(np.std(sica.S_[0, :]), 1)
    assert_almost_equal(np.std(sica.S_[1, :]), 1)
    assert_almost_equal(np.std(sica.S_[2, :]), 1)
