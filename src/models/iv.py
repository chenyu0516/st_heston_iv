import numpy as np
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt
from typing import Optional
from scipy.stats import norm


class HestonSmallTimeIV:
    def __init__(self, init_param: list, pp: Optional[float], pm: Optional[float]):
        """
        init_param: [rho, sigma, v0] where v0 is initial volatility (so variance is v0**2)
        pp, pm: Optional bounds for p
        """
        self.init_param = init_param
        limit_pm, limit_pp = self.p_bounds(init_param[0], init_param[1])
        if (pp is not None) and (pm is not None):
            self.pp = min(limit_pp, pp)
            self.pm = max(limit_pm, pm)
        else:
            self.pp = limit_pp
            self.pm = limit_pm
        # grid only for plotting; root finding will use proper bounds
        self.ps = np.linspace(self.pm + 1e-3, self.pp - 1e-3, 300)

    def Lambda(self, param, p):
        """
        Lambda(p) for Heston LDP, using variance y0 = v0**2
        """
        rho, sigma, v0 = param
        y0 = v0 ** 2
        rho_bar = np.sqrt(1 - rho ** 2)
        theta = 0.5 * sigma * p * rho_bar
        # Avoid singularity of cot(theta) at theta = n*pi
        eps = 1e-8
        # Safe theta for vectorized operations
        theta = np.where(np.abs(np.sin(theta)) < eps, theta + eps, theta)
        cot_theta = 1 / np.tan(theta)
        denominator = rho_bar * cot_theta - rho
        # Avoid zero denominator
        denominator = np.where(np.abs(denominator) < eps, np.nan, denominator)
        return y0 * p / (sigma * denominator)

    def Lambda_prime(self, param, p):
        """
        Derivative d/dp Lambda(p), see Eq (15) in Forde-Jacquier.
        """
        rho, sigma, v0 = param
        y0 = v0 ** 2
        rho_bar = np.sqrt(1 - rho ** 2)
        theta = 0.5 * sigma * p * rho_bar
        eps = 1e-8
        theta = np.where(np.abs(np.sin(theta)) < eps, theta + eps, theta)
        cot_theta = 1 / np.tan(theta)
        csc2 = 1 / (np.sin(theta) ** 2)
        denom = rho_bar * cot_theta - rho
        # Avoid zero denominator
        denom = np.where(np.abs(denom) < eps, np.nan, denom)
        # Lambda'(p)
        # First term
        A = y0 / (sigma * denom)
        # Second term
        B = (y0 * p * rho_bar * (0.5 * sigma) * (rho_bar**2 * csc2)) / (sigma * denom ** 2)
        return A + B

    def p_bounds(self, rho, sigma):
        """
        Compute p_minus, p_plus as in the Forde–Jacquier paper (Table after Eq (2)).
        """
        assert -1 < rho < 1, "rho must be in (-1, 1)"
        rho_bar = np.sqrt(1 - rho ** 2)
        denom = 0.5 * sigma * rho_bar
        if rho < 0:
            atan = np.arctan(rho_bar / rho)
            p_minus = atan / denom
            p_plus = (np.pi + atan) / denom
        elif rho > 0:
            atan = np.arctan(rho_bar / rho)
            p_minus = (-np.pi + atan) / denom
            p_plus = atan / denom
        else:  # rho == 0
            p_minus = -np.pi / sigma
            p_plus = np.pi / sigma
        return p_minus, p_plus

    def legendre_transform(self, param, x):
        """
        For a given x (log-moneyness), find p* such that x = Lambda'(p*),
        then compute Lambda^*(x) = p* x - Lambda(p*).
        Returns np.nan if solution not found.
        """
        pmin, pmax = self.p_bounds(param[0], param[1])
        # Avoid the singular endpoints
        def Lambda_p(p):
            return self.Lambda_prime(param, p) - x
        # Robust bracket for root finding
        try:
            sol = root_scalar(Lambda_p, bracket=(pmin + 1e-6, pmax - 1e-6), method='brentq')
        except Exception as e:
            # print(f"Root finding failed at x={x}: {e}")
            return np.nan
        if not sol.converged:
            # print(f"Root finding did not converge at x={x}")
            return np.nan
        pstar = sol.root
        Lval = self.Lambda(param, pstar)
        if np.isnan(Lval):
            return np.nan
        return pstar * x - Lval

    def IV_curve(self, param, lnK):
        """
        Compute the IV curve at log-moneyness values lnK.
        """
        # Vectorize legendre_transform over lnK
        rate_vals = np.array([self.legendre_transform(param, x) for x in lnK])
        # Only positive rate function is valid
        rate_vals = np.where(rate_vals > 0, rate_vals, np.nan)
        return np.abs(lnK) / np.sqrt(2 * rate_vals)

    def plot(self, param, lnK, iv):
        lnK_plot = np.linspace(self.pm + 1e-2, self.pp - 1e-2, 100)
        iv_fit = self.IV_curve(param, lnK_plot)
        plt.figure(figsize=(6, 4))
        plt.scatter(lnK, iv, s=25, label="Data", c="C0")
        plt.plot(lnK_plot, iv_fit, label="Small-time Heston IV fit", c="C1", lw=2)
        plt.xlabel(r"$\log(K/S_0)$")
        plt.ylabel("Implied Volatility")
        plt.legend()
        plt.tight_layout()
        plt.show()



def bls(s,k,r,t,sig):
    """
    Using black-schoss
    input:
      s: option price
      k: strike price
      r: risk-free interest
      t: DTE
      sig: implied vol

    out:
      call: call option price
      put: put option price
    """
    d1 = (np.log(s/k) + (r+sig**2/2)*t)/(sig*np.sqrt(t))
    d2 = (np.log(s/k) + (r-sig**2/2)*t)/(sig*np.sqrt(t))
    call = s * norm.cdf(d1) - k * np.exp(-r*t) * norm.cdf(d2)
    put = k * np.exp(-r*t) * norm.cdf(-d2) - s * norm.cdf(-d1)
    return call, put

"""
Bisection Method to find the call & put option price:
input: s,k,r,t,call/put
trying to bisection searching call/put implied volatility by seeing if the
result of calculation call/put option price by BS equation is near real value
"""
def bisection_call_iv(s,k,r,t,call):
    tol = 0.000001
    p1 = np.zeros(len(k))
    p2 = np.ones(len(k))*10
    for m in range(50):
        sig = (p1+p2)/2.0
        callbs = bls(s, k, r, t, sig)[0]
        if (abs(callbs-call) < tol).all():
            break
        index1 = callbs >= call
        index2 = callbs <= call
        p2[index1] = sig[index1]
        p1[index2] = sig[index2]
    return p1    # 隱含波動率

def bisection_put_iv(s,k,r,t,put):
    tol = 0.000001
    p1 = np.zeros(len(k))
    p2 = np.ones(len(k))*10
    for m in range(50):
        sig = (p1+p2)/2.0
        putbs = bls(s, k, r, t, sig)[1]
        if (abs(putbs-put) < tol).all():
            break
        index1 = putbs >= put
        index2 = putbs <= put
        p2[index1] = sig[index1]
        p1[index2] = sig[index2]
    return p1