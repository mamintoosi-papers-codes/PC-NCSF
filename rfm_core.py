"""
rfm_core.py — pure math/model building blocks for Protein-Conditional Riemannian
Flow Matching (PC-RFM).

Deliberately has NO argparse, NO dataset loading, and NO side effects at import
time, so it can be unit-tested (see test_rfm_core.py) and imported by both the
training script (rfm_protein_embedding_c.py) and any evaluation/report script
without re-running a training pipeline.

See the module docstring in rfm_protein_embedding_c.py for the conceptual
background (why flat-torus flow matching needs no exp/log maps, why the
training loss is not the NLL, etc.). This file only contains the mechanics.
"""

import math
import torch
import torch.nn as nn

TWO_PI = 2 * math.pi


def wrap_to_pi(x):
    """Shortest signed angular difference, mapped to (-pi, pi]."""
    return (x + math.pi) % TWO_PI - math.pi


def sample_conditional_path(x1, t, x0=None, generator=None):
    """
    Builds one sample from the conditional probability path p_t(x | x1).

    x1: (batch, 2) data angles (phi, psi)
    t:  (batch, 1) times in [0, 1]
    x0: (batch, 2) optional pre-drawn base points. If None, drawn fresh here
        (used during TRAINING, where fresh Monte Carlo noise every call is
        exactly what Flow Matching wants). Pass a fixed x0 (and a fixed t)
        during VALIDATION so the reported loss is comparable across epochs
        instead of being re-randomized every time.
    generator: optional torch.Generator, used only when x0 is None.

    Returns x_t (unwrapped, on the universal cover), target velocity u_t, x0.
    """
    if x0 is None:
        if generator is not None:
            x0 = torch.rand(x1.shape, generator=generator, device=x1.device) * TWO_PI
        else:
            x0 = torch.rand_like(x1) * TWO_PI
    delta = wrap_to_pi(x1 - x0)
    x_t = x0 + t * delta
    u_t = delta
    return x_t, u_t, x0


class VectorField(nn.Module):
    """Predicts dx/dt at (t, x) given the protein embedding c."""

    def __init__(self, cond_dim, hidden_dim, hidden_layers):
        super().__init__()
        in_dim = 4 + 1 + cond_dim  # [cos(phi), sin(phi), cos(psi), sin(psi), t, c]
        layers, d = [], in_dim
        for _ in range(hidden_layers):
            layers += [nn.Linear(d, hidden_dim), nn.SiLU()]
            d = hidden_dim
        layers += [nn.Linear(d, 2)]
        self.net = nn.Sequential(*layers)

    def forward(self, t, x, c):
        feats = torch.cat([torch.cos(x), torch.sin(x), t, c], dim=-1)
        return self.net(feats)


def cfm_loss(vf, embedding, x1, cond_labels, t=None, x0=None, generator=None):
    """
    Flow-matching regression loss. NOT the NLL (see module docstring).

    Training call (fresh noise every step):
        cfm_loss(vf, embedding, x1, cond_labels)

    Validation call (fixed noise, stable across epochs):
        cfm_loss(vf, embedding, x1, cond_labels, t=t_fixed, x0=x0_fixed)
    """
    batch = x1.shape[0]
    if t is None:
        if generator is not None:
            t = torch.rand(batch, 1, generator=generator, device=x1.device)
        else:
            t = torch.rand(batch, 1, device=x1.device)
    x_t, u_t, _ = sample_conditional_path(x1, t, x0=x0, generator=generator)
    c = embedding(cond_labels)
    v_pred = vf(t, x_t, c)
    return ((v_pred - u_t) ** 2).sum(dim=-1).mean()


def divergence_exact(vf, t, x, c):
    """Exact trace of the 2x2 Jacobian d(v)/d(x) (cheap: the manifold is 2-D)."""
    x = x.detach().requires_grad_(True)
    v = vf(t, x, c)
    div = torch.zeros(x.shape[0], device=x.device)
    for i in range(2):
        grad_i = torch.autograd.grad(v[:, i].sum(), x, retain_graph=True, allow_unused=True)[0]
        if grad_i is not None:
            div = div + grad_i[:, i]
    return div.detach()


def compute_nll(vf, embedding, x1, cond_labels, n_steps: int):
    """
    Per-sample -log p(x1 | c), via backward ODE integration (RK4, fixed step)
    and the instantaneous change-of-variables formula. See the derivation in
    rfm_protein_embedding_c.py's module docstring; the short version is:

        log p(x1|c) = log p_0(x0) - integral_0^1 Tr(dv/dz) dt

    with log p_0 constant (= -log((2*pi)^2)) because the base is uniform on
    the flat torus. Returns a (batch,) tensor -- one NLL value per residue.
    """
    dev = x1.device
    batch = x1.shape[0]
    c = embedding(cond_labels)

    def field_and_div(t_scalar, z):
        t = torch.full((batch, 1), t_scalar, device=dev)
        div = divergence_exact(vf, t, z, c)
        with torch.no_grad():
            v = vf(t, z, c)
        return v.detach(), div

    z = x1.clone()
    delta = torch.zeros(batch, device=dev)
    dt = -1.0 / n_steps
    t = 1.0
    for _ in range(n_steps):
        v1, d1 = field_and_div(t, z)
        v2, d2 = field_and_div(t + dt / 2, z + dt / 2 * v1)
        v3, d3 = field_and_div(t + dt / 2, z + dt / 2 * v2)
        v4, d4 = field_and_div(t + dt, z + dt * v3)
        z = z + (dt / 6) * (v1 + 2 * v2 + 2 * v3 + v4)
        delta = delta - (dt / 6) * (d1 + 2 * d2 + 2 * d3 + d4)
        t += dt

    log_p0 = -2.0 * math.log(TWO_PI)
    log_px1 = log_p0 - delta
    return -log_px1


def aggregate_per_protein_nll(nll_per_sample, cond_labels):
    """
    Groups per-residue NLL by protein id and averages WITHIN each protein.

    This mirrors exactly the convention used to build the paper's Table 4/5
    (confirmed from the simulation notebooks): for protein i,
        nll_i = -flow(c_i).log_prob(x_i).mean()
    i.e. the mean is taken across that protein's own residues first. Any
    further group- or dataset-level number (e.g. Table 2's single aggregate,
    or a per-group mean in Table 4) should be computed FROM these per-protein
    numbers, and whether that outer average is weighted by residue count or
    not depends on which table you are reproducing -- see the notes in
    rfm_protein_embedding_c.py. This function only does the inner stage.

    Returns: dict {protein_label (int): (mean_nll (float), n_residues (int))}
    """
    out = {}
    labels = cond_labels.detach().cpu()
    nll = nll_per_sample.detach().cpu()
    for pid in labels.unique().tolist():
        mask = labels == pid
        out[int(pid)] = (nll[mask].mean().item(), int(mask.sum().item()))
    return out
