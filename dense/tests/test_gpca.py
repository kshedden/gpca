import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import numpy as np
import pytest
from gpca import GPCA
from statsmodels.genmod import families
from numpy.testing import assert_allclose, assert_equal
import numdifftools as nd
import warnings

@pytest.mark.parametrize("miss_frac", [0, 0.5])
def test_score_gaussian(miss_frac):

    np.random.seed(23424)
    n, p = 100, 5

    for d in range(1, 5):
        icept = np.linspace(3, 5, p)
        fac = np.random.normal(size=(p, d))
        fac, _, _ = np.linalg.svd(fac, 0)
        sc = np.random.normal(size=(n, d))
        endog = np.dot(sc, fac.T) + icept
        valid = (np.random.uniform(size=(n, p)) > miss_frac).astype(np.bool)
        pca = GPCA(endog, d, valid=valid)

        par = np.concatenate((icept, fac.ravel()))
        grad = pca.score(par)
        ngrad = nd.Gradient(pca.loglike)(par)
        assert_allclose(grad, ngrad, rtol=1e-3, atol=1e-3)

@pytest.mark.parametrize("miss_frac", [0, 0.5])
def test_score_poisson(miss_frac):

    np.random.seed(23424)
    n, p = 100, 5

    for d in range(1, 5):

        # Generate the data
        icept = np.linspace(3, 5, p)
        fac = np.random.normal(size=(p, d))
        fac, _, _ = np.linalg.svd(fac, 0)
        sc = np.random.normal(size=(n, d))
        lp = np.dot(sc, fac.T) + icept
        mu = np.exp(lp)
        endog = np.random.poisson(mu, size=(n, p))
        valid = (np.random.uniform(size=(n, p)) > miss_frac).astype(np.bool)

        pca = GPCA(endog, d, valid=valid, family=families.Poisson())

        par = np.concatenate((icept, fac.ravel()))
        grad = pca.score(par)
        ngrad = nd.Gradient(pca.loglike)(par)
        assert_allclose(grad, ngrad, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("miss_frac", [0, 0.5])
def test_score_binomial(miss_frac):

    np.random.seed(23424)
    n, p = 100, 5

    for d in range(1, 5):

        # Generate the data
        icept = np.linspace(3, 5, p)
        fac = np.random.normal(size=(p, d))
        fac, _, _ = np.linalg.svd(fac, 0)
        sc = np.random.normal(size=(n, d))
        lp = np.dot(sc, fac.T) + icept
        mu = 1 / (1 + np.exp(-lp))
        endog = (np.random.uniform(size=(n, p)) < mu).astype(np.float64)
        valid = (np.random.uniform(size=(n, p)) > miss_frac).astype(np.bool)

        pca = GPCA(endog, d, family=families.Binomial(), valid=valid)

        par = np.concatenate((icept, fac.ravel()))
        grad = pca.score(par)
        ngrad = nd.Gradient(pca.loglike)(par)
        assert_allclose(grad, ngrad, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("miss_frac", [0, 0.5])
@pytest.mark.parametrize("d", [1, 2, 3, 4, 5])
def test_fit_gaussian(miss_frac, d):

    np.random.seed(23424)
    n, p = 2000, 5

    endog = np.random.normal(size=(n, p))
    f = np.arange(p, dtype=np.float64)
    endog += np.outer(np.random.normal(size=n), f)
    valid = (np.random.uniform(size=(n, p)) > miss_frac).astype(np.bool)
    pca = GPCA(endog, d, valid=valid)
    r = pca.fit()
    icept, fac = r.intercept, r.factors

    # GPCA fitted values
    resid0 = endog - icept
    fv = np.dot(resid0, np.dot(fac, fac.T)) + icept

    # The dominant factor should align with the only real factor.
    a = np.dot(f, r.factors[:, 0]) / np.sqrt(np.sum(f**2))
    assert_equal(np.abs(a) > [0.95, 0.99][miss_frac == 0], True)

    # If there are no missing values, we can compare to PCA.
    if miss_frac > 0:
        return

    # PCA fitted values
    icept1 = endog.mean(0)
    endogc = endog - icept1
    u, s, vt = np.linalg.svd(endogc, 0)
    s[d:] = 0
    fv0 = np.dot(u, np.dot(np.diag(s), vt)) + icept1
    fac1 = vt.T[:, 0:d]

    # Check that PCA and GPCA factors agree
    p1 = np.dot(fac, fac.T)
    p2 = np.dot(fac1, fac1.T)
    assert_allclose(np.trace(np.dot(p1, p2)), d, rtol=1e-10,
                    atol=1e-10)

    # Check that PCA and GPCA fitted values agree
    assert_allclose(fv, fv0, rtol=1e-10, atol=1e-10)

    # The scores should be centered at zero
    scores = pca.scores(r.params)
    assert_allclose(scores.mean(0), 0, rtol=1e-8, atol=1e-8)

    # The GPCA scores and PCA scores should agree up to
    # ordering.
    scores1 = u[:, 0:d] * s[0:d]
    c = np.corrcoef(scores.T, scores1.T)
    assert_allclose(np.abs(c).sum(0), 2*np.ones(2*d))

@pytest.mark.parametrize("miss_frac", [0, 0.5])
@pytest.mark.parametrize("d", [1, 2, 3, 4, 5])
def test_fit_poisson(miss_frac, d):

    np.random.seed(23424)
    n, p = 2000, 5

    icept = np.linspace(3, 5, p)
    fac = np.random.normal(size=(p, d))
    fac, _, _ = np.linalg.svd(fac, 0)
    sc = np.random.normal(size=(n, d))
    lp = np.dot(sc, fac.T) + icept
    mu = np.exp(lp)

    endog = np.random.poisson(mu, size=(n, p))
    valid = (np.random.uniform(size=(n, p)) > miss_frac).astype(np.bool)
    pca = GPCA(endog, d, valid=valid, family=families.Poisson())
    r = pca.fit()
    icept1, fac1 = r.intercept, r.factors

    # Check intercepts versus population values
    if not np.allclose(icept, icept1, atol=1e-2, rtol=1e-1):
        warnings.warn("icept=%s icept1=%s" % (icept, icept1))

    # Check factors versus population values
    p1 = np.dot(fac, fac.T)
    p2 = np.dot(fac1, fac1.T)
    if not np.allclose(np.trace(np.dot(p1, p2)), d, rtol=[1e-2, 0.05][miss_frac > 0]):
        warnings.warn("d=%s, trace=%s" % (d, np.trace(np.dot(p1, p2))))

    # Scores should be approximately centered
    scores = pca.scores(r.params)
    if not np.allclose(scores.mean(), 0, atol=[1e-2, 1e-1][miss_frac > 0]):
        warnings.warn(str(scores.mean(0)))

    assert(r.score_norm < 0.01)
