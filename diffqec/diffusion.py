"""Discrete diffusion kernels for binary data."""
from typing import Tuple
import numpy as np
import torch
import torch.nn.functional as F


def cosine_schedule(T: int, s: float = 0.008) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cosine noise schedule (Nichol & Dhariwal).

    Returns
    -------
    alpha_bar : np.ndarray shape (T+1,)  — ᾱ_t for t=0..T
    beta      : np.ndarray shape (T,)    — β_t for t=1..T
    alpha     : np.ndarray shape (T,)    — α_t for t=1..T
    """
    steps = np.arange(T + 1, dtype=np.float64)
    # ᾱ_t = cos²(((t/T + s)/(1+s)) · π/2) / cos²((s/(1+s)) · π/2)
    num = np.cos(((steps / T + s) / (1 + s)) * np.pi / 2) ** 2
    den = np.cos((s / (1 + s)) * np.pi / 2) ** 2
    alpha_bar = num / den
    alpha_bar[0] = 1.0  # clean at t=0

    alpha = alpha_bar[1:] / alpha_bar[:-1]
    beta = 1.0 - alpha
    beta = np.clip(beta, 1e-4, 0.9999)
    # recompute alpha/alpha_bar after clipping
    alpha = 1.0 - beta
    alpha_bar = np.concatenate([[1.0], np.cumprod(alpha)])
    return alpha_bar.astype(np.float32), beta.astype(np.float32), alpha.astype(np.float32)


def sample_xt(x0: torch.Tensor, t: torch.Tensor, alpha_bar: np.ndarray) -> torch.Tensor:
    """Forward sample: q(x_t | x_0) for binary symmetric diffusion.

    Parameters
    ----------
    x0       : (batch, L) bool or float {0,1}
    t        : diffusion timestep (1 .. T), scalar int or (batch,) Tensor
    alpha_bar: from cosine_schedule(T)

    Returns
    -------
    xt : (batch, L) float {0,1}
    """
    if isinstance(t, int):
        t = torch.tensor([t], device=x0.device)
    ab = torch.from_numpy(alpha_bar).to(x0.device)[t.long()]  # scalar or (batch,)
    flip_prob = (1.0 - ab) / 2.0
    if ab.dim() > 0:
        flip_prob = flip_prob.view(-1, 1)
    # flip each bit independently with probability (1 - ᾱ_t)/2
    mask = torch.rand_like(x0.float()) < flip_prob
    xt = torch.where(mask, 1.0 - x0.float(), x0.float())
    return xt


def reverse_posterior_prob(xt: torch.Tensor, x0: torch.Tensor,
                           beta_t: float, alpha_bar_t: float,
                           alpha_bar_tm1: float) -> torch.Tensor:
    """Compute P(x_{t-1}=1 | x_t, x_0) for a single bit.

    Uses the closed-form binary-symmetric posterior.

    Parameters
    ----------
    xt          : (...,) float {0,1}
    x0          : (...,) float {0,1}
    beta_t      : float
    alpha_bar_t : float
    alpha_bar_tm1: float

    Returns
    -------
    prob : (...,) float  — probability that x_{t-1} = 1
    """
    # q(x_t | x_{t-1}) — flip with prob beta_t/2
    q_b_given_a = lambda b, a: (beta_t / 2.0) if b != a else (1.0 - beta_t / 2.0)
    # q(x_{t-1} | x_0) — flip with prob (1 - alpha_bar_{t-1})/2
    q_a_given_c = lambda a, c: ((1.0 - alpha_bar_tm1) / 2.0) if a != c else ((1.0 + alpha_bar_tm1) / 2.0)
    # q(x_t | x_0) — flip with prob (1 - alpha_bar_t)/2
    q_b_given_c = lambda b, c: ((1.0 - alpha_bar_t) / 2.0) if b != c else ((1.0 + alpha_bar_t) / 2.0)

    # Vectorised over tensors
    # P(x_{t-1}=1 | x_t, x_0) = q(x_t | x_{t-1}=1) * q(x_{t-1}=1 | x_0) / q(x_t | x_0)
    # To avoid 0/0 when all probs line up, add epsilon
    eps = 1e-12

    def _qbgivena(b, a):
        return torch.where(torch.abs(b - a) < 0.5,
                           torch.full_like(b, 1.0 - beta_t / 2.0 + eps),
                           torch.full_like(b, beta_t / 2.0 + eps))

    def _qagivenc(a, c):
        return torch.where(torch.abs(a - c) < 0.5,
                           torch.full_like(a, (1.0 + alpha_bar_tm1) / 2.0 + eps),
                           torch.full_like(a, (1.0 - alpha_bar_tm1) / 2.0 + eps))

    def _qbgivenc(b, c):
        return torch.where(torch.abs(b - c) < 0.5,
                           torch.full_like(b, (1.0 + alpha_bar_t) / 2.0 + eps),
                           torch.full_like(b, (1.0 - alpha_bar_t) / 2.0 + eps))

    num = _qbgivena(xt, torch.ones_like(xt)) * _qagivenc(torch.ones_like(x0), x0)
    den = _qbgivenc(xt, x0)
    prob = num / den
    return torch.clamp(prob, 0.0, 1.0)


def sample_x_prev(xt: torch.Tensor, x0_prob: torch.Tensor,
                  beta_t: float, alpha_bar_t: float,
                  alpha_bar_tm1: float) -> torch.Tensor:
    """Sample x_{t-1} from p(x_{t-1} | x_t) by marginalising over x_0.

    Parameters
    ----------
    xt           : (batch, L) float {0,1}
    x0_prob      : (batch, L) float — P(x_0 = 1) from model
    beta_t       : float
    alpha_bar_t  : float
    alpha_bar_tm1: float

    Returns
    -------
    x_prev : (batch, L) float {0,1}
    """
    # Marginalise: P(x_{t-1}=1 | x_t) = Σ_{c∈{0,1}} P(x_{t-1}=1 | x_t, x_0=c) * P(x_0=c)
    p1_given_0 = reverse_posterior_prob(xt, torch.zeros_like(x0_prob), beta_t, alpha_bar_t, alpha_bar_tm1)
    p1_given_1 = reverse_posterior_prob(xt, torch.ones_like(x0_prob), beta_t, alpha_bar_t, alpha_bar_tm1)
    p1 = p1_given_0 * (1.0 - x0_prob) + p1_given_1 * x0_prob
    return (torch.rand_like(p1) < p1).float()


def get_x0_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Argmax over the 2-class logits to recover binary x_0.

    logits: (batch, L, 2)
    returns: (batch, L) float {0,1}
    """
    return logits.argmax(dim=-1).float()
