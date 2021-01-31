import numpy as np
import statsmodels
import statsmodels.base.model as base
from statsmodels.genmod import families
import warnings
from scipy.optimize import minimize, minimize_scalar

class GPCA(base.LikelihoodModel):

    def __init__(self, endog, ndim, valid=None, family=None, penmat=None):
        """
        Fit a generalized principal component analysis.

        This analysis fits a generalized linear model (GLM) to a
        rectangular data array.  The linear predictor, which in a GLM
        would be derived from covariates, is instead represented as a
        factor-structured matrix.  If endog is n x p and we wish to
        extract d factors, then the linear predictor is represented as
        1*icept' + (s - 1*icept')*F*F', where 1 is a column vector of
        n 1's, s is a n x p matrix containing the 'saturated' linear
        predictor, and F is a p x d orthogonal matrix of loadings.

        Parameters
        ----------
        endog : array-like
            The data to which a reduced-rank structure is fit.
        ndim : integer
            The dimension of the low-rank structure.
        family : GLM family instance
            The GLM family to use in the analysis
        valid : Boolean array, shaped like endog
            endog is only observed where valid is True

        Returns
        -------
        A GPCAResults instance.

        Notes
        -----
        Estimation uses the Grassmann optimization approach of Edelman,
        rather than the approaches from Landgraf and Lee.

        References
        ----------
        A. Landgraf, Y.Lee (2019). Generalized Principal Component Analysis:
        Projection of saturated model parameters.  Technometrics.
        https://www.asc.ohio-state.edu/lee.2272/mss/tr890.pdf

        Edelman,Arias, Smith (1999).  The geometry of algorithms with orthogonality
        constraints.
        https://arxiv.org/abs/physics/9806030
        """

        if family is None:
            # Default family
            family = families.Gaussian()

        self.family = family
        self.endog = np.asarray(endog)
        self.valid = valid
        self.ndim = ndim

        if penmat is not None:
            pm = []
            if len(penmat) != 2:
                msg = "penmat must be a tuple of length 2"
                raise ValueError(msg)
            for j in range(2):
                if np.isscalar(penmat[j]):
                    n, p = endog.shape
                    pm.append(self._gen_penmat(penmat[j], n, p))
                else:
                    pm.append(penmat[j])
            self.penmat = pm

        # Calculate the saturated parameter
        if isinstance(family, families.Poisson):
            self.satparam = np.where(self.endog > 0, self.endog, np.exp(-3))
            self.satparam = np.log(self.satparam)
        elif isinstance(family, families.Binomial):
            self.satparam = np.where(self.endog == 1, 3, -3)
        elif isinstance(family, families.Gaussian):
            self.satparam = self.endog
        else:
            raise ValueError("Unknown family")


    def _gen_penmat(self, f, n, p):
        pm = np.zeros((p-2, p))
        for k in range(p-2):
            pm[k, k:k+3] = [-1, 2, -1]
        return f * pm * n

    def _linpred(self, params):

        n, p = self.endog.shape

        icept = params[0:p]
        qm = params[p:].reshape((p, self.ndim))

        resid = self.satparam - icept

        if self.valid is not None:
            resid *= self.valid

        lp = icept + np.dot(np.dot(resid, qm), qm.T)

        if self.valid is not None:
            lp *= self.valid

        return icept, qm, resid, lp

    def _flip(self, params):
        """
        Multiply factors by -1 so that the majority of entries
        are positive.
        """
        p = self.endog.shape[1]
        icept = params[0:p]
        qm = params[p:].reshape((p, self.ndim))
        for j in range(self.ndim):
            if np.sum(qm[:, j] < 0) > np.sum(qm[:, j] > 0):
                qm[:, j] *= -1
        return np.concatenate((icept, qm.ravel()))

    def predict(self, params, linear=False):
        """
        Return the fitted mean or its linear predictor.

        Parameters
        ----------
        params : array-like
            The parameters to use to produce the fitted mean
        linear : boolean
            If true, return the linear predictor, otherwise
            return the fitted mean, which is the inverse
            link function evaluated at the linear predictor.

        Returns an array with the same shape as endog, containing
        fitted values corresponding to the given parameters.
        """

        _, _, _, lp = self._linpred(params)

        if linear:
            return lp
        else:
            return self.family.fitted(lp)


    def scores(self, params):
        """
        Returns the PC scores for each case.

        Parameters
        ----------
        params : array-like
            The parameters at which the scores are
            calculated.

        Returns
        -------
        An array of scores.
        """

        _, qm, resid, _ = self._linpred(params)

        return np.dot(resid, qm)


    def loglike(self, params):

        icept, qm, _, lp = self._linpred(params)
        expval = self.family.link.inverse(lp)

        if self.valid is None:
            ll = self.family.loglike(self.endog.ravel(), expval.ravel())
        else:
            ii = np.flatnonzero(self.valid.ravel())
            ll = self.family.loglike(self.endog.ravel()[ii],
                                     expval.ravel()[ii])

        if hasattr(self, "penmat"):
            pm = self.penmat
            ll -= np.sum(np.dot(pm[0], icept)**2)
            for j in range(self.ndim):
                ll -= np.sum(np.dot(pm[1], qm[:, j])**2)

        return ll

    def _orthog(self, qm, v):
        for i in range(5):
            v -= np.dot(qm, np.dot(qm.T, v))
            u = np.max(np.abs(np.dot(qm.T, v)))
            if u < 1e-10:
                break
        return v


    def score(self, params, project=False):

        icept, qm, resid, lp = self._linpred(params)

        # The fitted means
        mu = self.family.fitted(lp)

        # The derivative of the log-likelihood with respect to
        # the canonical parameters.
        # TODO: usually the link.deriv and variance will cancel
        sf = (self.endog - mu) / self.family.link.deriv(mu)
        sf /= self.family.variance(mu)
        if self.valid is None:
            si = (sf - np.dot(np.dot(sf, qm), qm.T)).sum(0)
        else:
            sf *= self.valid
            si = (sf - np.dot(np.dot(sf, qm), qm.T))
            si = (si * self.valid).sum(0)

        # The score with respect to the factors
        rts = np.dot(resid.T, sf)

        df = np.dot(rts, qm) + np.dot(rts.T, qm)

        if hasattr(self, "penmat"):
            pm = self.penmat
            si -= 2 * np.dot(pm[0].T, np.dot(pm[0], icept))
            for j in range(self.ndim):
                df[:, j] -= 2 * np.dot(pm[1].T, np.dot(pm[1], qm[:, j]))

        if project:
            df = self._orthog(qm, df)

        sc = np.concatenate((si, df.ravel()))

        return sc

    def _update_icept(self, pa):

        _, p = self.endog.shape
        d = self.ndim
        fac = pa[p:]

        def fun(x):
            pa = np.concatenate((x, fac))
            return -self.loglike(pa)

        def jac(x):
            pa = np.concatenate((x, fac))
            return -self.score(pa)[0:p]

        mm = minimize(fun, pa[0:p], method="BFGS", jac=jac)
        fail = not mm.success
        pa = np.concatenate((mm.x, fac))

        return pa, fail, mm.jac

    def _update_factors_cg(self, pa, maxiter=100, gtol=1e-5):

        _, p = self.endog.shape
        d = self.ndim
        icept = pa[0:p]

        # Initialization
        y0 = pa[p:].reshape((p, d))
        g0 = -self.score(pa)[p:].reshape((p, d))
        g0 -= np.dot(y0, np.dot(y0.T, g0))
        if np.sqrt(np.sum(g0**2)) < gtol:
            return pa, False, g0
        h = -g0

        for itr in range(maxiter):

            u, s, vt = np.linalg.svd(h, 0)
            v = vt.T

            def f(t):
                co = np.cos(s*t)
                si = np.sin(s*t)
                qq = np.dot(y0, np.dot(v, co[:, None] * vt)) + np.dot(u, si[:, None] * vt)
                qq, _ = np.linalg.qr(qq) # Orthogonality degrades due to rounding
                qp = np.concatenate((icept, qq.ravel()))
                return qp

            mm = minimize_scalar(lambda x: -self.loglike(f(x)))
            t = mm.x
            if not mm.success or (np.abs(t) < 1e-10):
                return pa, True, g0
            pa = f(t)
            y1 = pa[p:].reshape((p, d))
            g1 = -self.score(pa)[p:].reshape((p, d))
            g1 -= np.dot(y1, np.dot(y1.T, g1))
            if np.sqrt(np.sum(g1**2)) < gtol:
                return pa, False, g1

            # Parallel transport h
            co = np.cos(s*t)
            si = np.sin(s*t)
            th = -np.dot(y0, v * si) + u * co
            th = np.dot(th, s[:, None] * vt)

            # Parallel transport g
            tg = np.dot(y0, v * si) + u * (1 - co)
            tg = np.dot(tg, np.dot(u.T, g0))
            tg = g0 - tg

            ga = np.trace(np.dot((g1 - tg).T, g1)) / np.trace(np.dot(g0.T, g0))
            if (itr + 1) % (p * (p - d)) == 0:
                h = -g1
            else:
                h = -g1 + ga * th
            g0, y0 = g1, y1

        return pa, False, g1

    def fit(self, maxiter=10, gtol=1e-8):

        n, p = self.endog.shape
        d = self.ndim

        # Starting values
        if self.valid is None:
            icept = self.satparam.mean(0)
            cnp = self.satparam - icept
        else:
            icept = (self.satparam * self.valid).sum(0) / self.valid.sum(0)
            cnp = self.satparam - icept
            cnp = np.where(self.valid, cnp, np.outer(np.ones(n), icept))

        _, _, vt = np.linalg.svd(cnp, 0)
        v = vt.T
        v = v[:, 0:d]
        pa = np.concatenate((icept, v.ravel()))
        converged = False

        for itr in range(maxiter):

            pa, fail1, g1 = self._update_icept(pa)
            pa, fail2, g2 = self._update_factors_cg(pa)

            gn = np.sqrt(np.sum(g1**2) + np.sum(g2**2))
            if gn < gtol:
                converged = True
                break

            if fail1 and fail2:
                break

        if not converged:
            warnings.warn("GPCA did not converge, |G|=%f" % gn)

        pa = self._flip(pa)
        icept = pa[0:p]
        qm = pa[p:].reshape((p, d))

        results = GPCAResults(icept, qm)
        results.converged = converged
        results.params = pa
        results.score_norm = gn

        results._iterations = itr
        return results


class GPCAResults:

    def __init__(self, intercept, factors):

        self.intercept = intercept
        self.factors = factors
