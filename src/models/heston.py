import numpy as np

def simulate_heston(
    S0=100, V0=0.04, T=1.0, dt=1/252,
    mu=0.0, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.7,
    rng=None
):
    if rng is None:
        rng = np.random.default_rng()  # safe default

    N = int(T / dt)
    t = np.linspace(0, T, N+1)

    S = np.zeros(N+1)
    V = np.zeros(N+1)
    S[0] = S0
    V[0] = V0

    for i in range(N):
        z1 = rng.normal()
        z2 = rho * z1 + np.sqrt(1 - rho**2) * rng.normal()

        V[i+1] = np.abs(V[i] + kappa * (theta - V[i]) * dt + sigma * np.sqrt(V[i]) * np.sqrt(dt) * z2)
        S[i+1] = S[i] * np.exp((mu - 0.5 * V[i]) * dt + np.sqrt(V[i] * dt) * z1)

    return t, S, V

class Heston:
    def __init__(self, mu, kappa, theta, sigma, rho):
        self.mu = mu
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.rho = rho
        
    def CF(self, u, v, t, S0):
        """
        closed-form characteristic function from Heston(1993)
        f(v, t, S0 ;u)
        v: spot volitility
        t: time to maturity
        S0: spot price
        """
        return self.CF_no_osci(u, v, t)*np.exp(1j*u*np.log(S0))
    
    def CF_no_osci(self, u, v, t):
        """
        closed-form characteristic function from Heston(1993)
        f(v, t, S0 ;u)
        v: spot volitility
        t: time to maturity
        S0: spot price
        
        This the the function provide another form
        E[e^(iu ln(S_t))]
        """
        # Same formula, but with explicit real/imag handling to avoid trap
        d = np.sqrt((self.rho * self.sigma * 1j * u - self.kappa)**2 - self.sigma**2 * (-1j * u - u**2))
        # Use np.where to pick branch for d if vectorized, here scalar is fine
        g = (self.kappa - self.rho * self.sigma * 1j * u + d) / (self.kappa - self.rho * self.sigma * 1j * u - d)
        exp1 = np.exp(1j * u * self.mu * t)
        exp2 = np.exp((self.kappa * self.theta / self.sigma**2) * ((self.kappa - self.rho * self.sigma * 1j * u + d) * t\
            - 2.0 * np.log((1 - g * np.exp(d * t)) / (1 - g))))
        exp3 = np.exp(v / self.sigma**2 * (self.kappa - self.rho * self.sigma * 1j * u + d) * \
            (1 - np.exp(d * t)) / (1 - g * np.exp(d * t)))
        
        return exp1 * exp2 * exp3