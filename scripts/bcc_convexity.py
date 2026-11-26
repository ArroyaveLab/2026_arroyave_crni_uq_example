"""BCC convexity: d2G/dx2 of BCC_A2 (one sublattice, RK L0 + L1, each a+bT) and a one-sided penalty at observed states.

G''(x,T) = RT/(x(1-x)) + d2Gex/dx2, x = x_Ni, Gex = x(1-x)[L0 + L1(1-2x)]  (pycalphad: (y_CR - y_NI) = 1 - 2x).
The pure-element lattice stabilities are linear in x and do not enter G''.
"""
import numpy as np

R = 8.314462618


def gex_dd(L0, L1, x):
    # Gex = L0 (x - x^2) + L1 (x - x^2)(1 - 2x) = L0 (x - x^2) + L1 (x - 3x^2 + 2x^3)
    return L0 * (-2.0) + L1 * (-6.0 + 12.0 * x)


def curvature(theta, T, x):
    L0 = theta[2] + theta[3] * T
    L1 = theta[4] + theta[5] * T
    return R * T / (x * (1 - x)) + gex_dd(L0, L1, x)


def dcurv_dtheta(T, x):
    g = np.zeros(10)
    g[2], g[3] = -2.0, -2.0 * T
    g[4], g[5] = (-6.0 + 12.0 * x), (-6.0 + 12.0 * x) * T
    return g


def tc_estimate(L0a, L0b, L1a, L1b, xs=np.linspace(0.02, 0.98, 97)):
    """highest T at which G'' <= 0 somewhere (spinodal top), scanning T."""
    for T in np.arange(4000.0, 100.0, -1.0):
        c = R * T / (xs * (1 - xs)) + gex_dd(L0a + L0b * T, L1a + L1b * T, xs)
        if np.any(c <= 0):
            return float(T), float(xs[np.argmin(c)])
    return None, None


class ConvexityPenalty:
    """P = 0.5 * sum (E_def/sigma)^2 over observed single-phase BCC states with G'' < 0,
    E_def = (-G'') * w^2 / 8: the energy by which a chord of width w lies below the BCC curve when
    its curvature is -G'' (sagitta). w = 0.1 (mole fraction), sigma = 1000 J/mol-atom (hinge sigma)."""

    def __init__(self, states, w=0.1, sigma=1000.0):
        self.states = [(float(T), float(x)) for T, x in states]
        self.k = w * w / 8.0
        self.sigma = sigma

    def __call__(self, theta):
        P, G = 0.0, np.zeros(10)
        for T, x in self.states:
            c = curvature(theta, T, x)
            if c < 0:
                e = -c * self.k / self.sigma
                P += 0.5 * e * e
                G += e * (-self.k / self.sigma) * dcurv_dtheta(T, x)
        return P, G

    def violations(self, theta):
        return [(T, x, curvature(theta, T, x)) for T, x in self.states if curvature(theta, T, x) < 0]
