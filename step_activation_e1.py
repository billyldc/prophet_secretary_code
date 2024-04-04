import numpy as np
from numpy import exp
from scipy.optimize import minimize, NonlinearConstraint, LinearConstraint
from tqdm import tqdm
from multiprocess import Pool
import sys


def approx(x: float) -> float:
    """Upper bound (e^x - 1 - x)/x^2 for x > 0 without numerical instability."""
    if x < 1e-3:
        return 1 / 2 + x / 6 + x**2 / 24 + x**3 / 24
    else:
        return (exp(x) - 1 - x) / x**2


def optimize_two_stage(s, x0, h0, h_ot):

    def objective(args: np.ndarray) -> float:
        Gamma = args[0]
        return -Gamma

    def constr_ratio(args: np.ndarray) -> np.ndarray:
        Gamma, a, b, c = args

        ea = exp(-a * h_ot)
        eb = exp(-b * h_ot)
        ec = exp(-c * h_ot - s * (c - b) * (1 - x0 - h_ot))
        e1 = exp(-h_ot - s * (1 - b) * (1 - x0 - h_ot))

        w = h_ot    # slope of K(t)
        I0 = (1 - ea) / w
        J0 = I0

        t = h0      # slope of L(t)
        I1 = (ea - eb) / w
        J1 = I1 - t * eb * (b-a)**2 * approx((b-a)*h_ot)

        w = h_ot + s * (1 - x0 - h_ot)
        I2 = (eb - ec) / w
        J2 = I2 - t * ec * (c-b)**2 * approx((c-b)*h_ot)

        t = h0 + s * (x0 - h0)
        I3 = (ec - e1) / w
        J3 = I3 - t * e1 * (1-c)**2 * approx((1-c)*h_ot)
        R1 = J0 + J1 + J2 + J3
        R2 = s * (J2 + J3)
        R3 = I1 + I2 + I3
        R4 = s * I3

        return [R1 - Gamma, R2 - Gamma, R3 - Gamma, R4 - Gamma]

    res = minimize(
        objective,
        x0=[0.5, 0.1, 0.2, 0.3],
        bounds=[(0.5, 1.), (0.0, 0.5), (0.0, 0.6), (0.0, 1.0)],
        constraints=[
            LinearConstraint([0, -1, 1, 0], lb=0., ub=np.inf),  # b >= a
            LinearConstraint([0, 0, -1, 1], lb=0., ub=np.inf),  # c >= b
            NonlinearConstraint(constr_ratio, lb=0., ub=np.inf),
        ],
    )

    return -res.fun


def compute_h(x: float, s: float) -> float:

    def objective(args) -> float:
        t = args[0]
        y = t
        obj = t

        while s * (1 - y) > 1:
            obj = obj + (s * x - x / (1 - y)) / (s - 1)
            y = y + x

        return -obj

    res = minimize(objective, x0=[x / 2], bounds=[(0, x)])

    return -res.fun


def compute_ratio(x0: float, h0: float, eps) -> float:

    # Use the following choices of s for the 0.688 ratio
    # if x0 < 0.35:
    #     s = 3.
    # elif x0 < 0.6:
    #     s = 2.5
    # else:
    #     s = 2.

    # Use this fixed choice of s for the 0.686 ratio
    s = 2.

    h = compute_h(x0, s)
    h_ot = min(h - h0, 1 - x0)
    ratio = optimize_two_stage(s, x0, h0, h_ot)
    error = s * (1.5 * s + 0.5) * eps
    return x0, h0, ratio, error


if __name__ == "__main__":
    n = int(sys.argv[1])

    eps = 1 / n

    map_range = [(x0, h0, eps) for x0 in np.arange(eps, 1, eps)
                 for h0 in np.arange(0., x0, eps)]

    with Pool() as pool:
        res = list(
            tqdm(
                pool.imap_unordered(
                    lambda args: compute_ratio(args[0], args[1], args[2]),
                    map_range,
                ),
                total=len(map_range),
            )
        )

    print(f"Ratio = {min([ratio-error for _, _, ratio, error in res]): .4f}")
