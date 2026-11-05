"""Euclidean projection onto sum(beta)=1, beta>=beta_min.

The degenerate feasible case N*beta_min == 1 is supported. Float64 output.
"""
import numpy as np


def project_allocation(proposal, beta_min=.01):
    v = np.asarray(proposal, dtype=np.float64)
    if v.ndim != 1 or not v.size or not np.isfinite(v).all():
        raise ValueError('Expected a nonempty finite vector')
    if not np.isfinite(beta_min) or beta_min < 0 or v.size*beta_min > 1:
        raise ValueError('Infeasible lower bound')
    mass = 1.-v.size*beta_min
    if mass == 0:
        return np.full(v.size, beta_min)
    # Translation invariance permits subtracting max before sorting. Values
    # more than mass below max cannot be active; clipping these avoids overflow
    # for opposite-sign values near float64 limits without changing projection.
    with np.errstate(over='ignore'):
        shifted = v-v.max()
    shifted = np.maximum(shifted, -mass)
    # Project v-beta_min; subtracting a common constant gives the same result.
    u = np.sort(shifted)[::-1]
    cssv = np.cumsum(u)-mass
    active = np.flatnonzero(u-cssv/np.arange(1, v.size+1) > 0)
    rho = active[-1]
    theta = cssv[rho]/(rho+1)
    return beta_min+np.maximum(shifted-theta, 0.)
