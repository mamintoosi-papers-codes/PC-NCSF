"""
test_rfm_core.py — unit tests for rfm_core.py.

Run with either:
    pytest test_rfm_core.py -v
or:
    python test_rfm_core.py

Two kinds of checks:
  1. Analytic checks with a KNOWN closed-form answer (no training involved) --
     these catch sign errors / formula bugs in compute_nll itself.
  2. A learning sanity check -- confirms cfm_loss and compute_nll move in the
     same (correct) direction as the model actually learns something, on a
     synthetic dataset where we know the truth by construction.
These are exactly the checks used to validate the implementation before
handing it over.
"""

import math
import torch
import torch.nn as nn

from rfm_core import (
    TWO_PI,
    wrap_to_pi,
    sample_conditional_path,
    VectorField,
    cfm_loss,
    compute_nll,
    aggregate_per_protein_nll,
)


def test_wrap_to_pi_basic():
    x = torch.tensor([0.1, 6.2, -6.2, 3 * math.pi])
    w = wrap_to_pi(x)
    assert torch.all(w > -math.pi - 1e-6) and torch.all(w <= math.pi + 1e-6)
    # 3*pi wraps to -pi (or +pi, both are the same point on the circle)
    assert abs(abs(w[3].item()) - math.pi) < 1e-4


def test_cfm_loss_is_finite_and_differentiable():
    torch.manual_seed(0)
    batch, cond_dim, n_proteins = 32, 8, 10
    x1 = torch.rand(batch, 2) * TWO_PI
    labels = torch.randint(0, n_proteins, (batch,))
    embedding = nn.Embedding(n_proteins, cond_dim)
    vf = VectorField(cond_dim, hidden_dim=32, hidden_layers=2)

    loss = cfm_loss(vf, embedding, x1, labels)
    assert torch.isfinite(loss)
    loss.backward()
    assert vf.net[0].weight.grad is not None
    assert embedding.weight.grad is not None


def test_nll_shape_and_finiteness():
    torch.manual_seed(0)
    batch, cond_dim, n_proteins = 16, 8, 5
    x1 = torch.rand(batch, 2) * TWO_PI
    labels = torch.randint(0, n_proteins, (batch,))
    embedding = nn.Embedding(n_proteins, cond_dim)
    vf = VectorField(cond_dim, hidden_dim=32, hidden_layers=2)

    nll = compute_nll(vf, embedding, x1, labels, n_steps=15)
    assert nll.shape == (batch,)
    assert torch.isfinite(nll).all()


def test_nll_analytic_zero_field():
    """
    If the vector field is (numerically) zero everywhere, the ODE map is the
    identity: x0 == x1 for every sample. Since the base distribution is
    uniform on T^2 with constant density 1/(2*pi)^2, the NLL must then equal
    exactly 2*log(2*pi) for EVERY sample, regardless of what x1 is.
    This is the single most important check: it validates the sign
    conventions in the change-of-variables integral in compute_nll.
    """
    class ZeroVF(nn.Module):
        def forward(self, t, x, c):
            return x * 0.0  # stays in the autograd graph (grad = 0), unlike torch.zeros_like

    torch.manual_seed(0)
    batch, cond_dim, n_proteins = 20, 4, 3
    x1 = torch.rand(batch, 2) * TWO_PI
    labels = torch.randint(0, n_proteins, (batch,))
    embedding = nn.Embedding(n_proteins, cond_dim)

    nll = compute_nll(ZeroVF(), embedding, x1, labels, n_steps=10)
    expected = 2.0 * math.log(TWO_PI)
    assert torch.allclose(nll, torch.full_like(nll, expected), atol=1e-4), (
        f"expected {expected} for all samples, got {nll.tolist()}"
    )


def test_training_reduces_nll_on_concentrated_synthetic_clusters():
    """
    Trains a tiny PC-RFM on 3 synthetic 'proteins', each a tight cluster of
    angles. A correct implementation must show:
      (a) cfm_loss decreases over training,
      (b) NLL drops from close to the uniform-base value (~3.68) to a much
          lower (more negative) value, since the data is now highly
          concentrated (high density => low/negative NLL).
    This does not prove numerical exactness (only the analytic zero-field
    test does that), but it validates the full pipeline end-to-end.
    """
    torch.manual_seed(1)
    n_proteins = 3
    centers = torch.tensor([[0.5, 0.5], [3.0, 4.0], [5.5, 1.0]])

    def sample_batch(n=256):
        labels = torch.randint(0, n_proteins, (n,))
        noise = torch.randn(n, 2) * 0.05
        x1 = (centers[labels] + noise) % TWO_PI
        return x1, labels

    embedding = nn.Embedding(n_proteins, 8)
    vf = VectorField(8, hidden_dim=64, hidden_layers=2)
    opt = torch.optim.Adam(list(vf.parameters()) + list(embedding.parameters()), lr=3e-3)

    x1_eval, labels_eval = sample_batch(512)
    nll_before = compute_nll(vf, embedding, x1_eval, labels_eval, n_steps=20).mean().item()

    losses = []
    for step in range(400):
        x1b, labelsb = sample_batch(128)
        opt.zero_grad()
        loss = cfm_loss(vf, embedding, x1b, labelsb)
        loss.backward()
        opt.step()
        losses.append(loss.item())

    nll_after = compute_nll(vf, embedding, x1_eval, labels_eval, n_steps=20).mean().item()

    assert losses[-1] < losses[0], "cfm_loss did not decrease during training"
    assert nll_after < nll_before - 1.0, (
        f"NLL should drop substantially on concentrated clusters "
        f"(before={nll_before:.3f}, after={nll_after:.3f})"
    )


def test_aggregate_per_protein_nll_matches_manual_grouping():
    torch.manual_seed(0)
    nll = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    labels = torch.tensor([0, 0, 1, 1, 1, 2])
    result = aggregate_per_protein_nll(nll, labels)

    assert result[0] == (1.5, 2)          # mean(1,2), count 2
    assert abs(result[1][0] - 4.0) < 1e-9  # mean(3,4,5)
    assert result[1][1] == 3
    assert result[2] == (6.0, 1)


if __name__ == "__main__":
    tests = [
        test_wrap_to_pi_basic,
        test_cfm_loss_is_finite_and_differentiable,
        test_nll_shape_and_finiteness,
        test_nll_analytic_zero_field,
        test_training_reduces_nll_on_concentrated_synthetic_clusters,
        test_aggregate_per_protein_nll_matches_manual_grouping,
    ]
    for fn in tests:
        fn()
        print(f"PASSED: {fn.__name__}")
    print(f"\nAll {len(tests)} tests passed.")
