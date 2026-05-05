"""
============================================================
Iterative Solve: CG, LSMR, and Polynomial Preconditioning
============================================================

This example demonstrates the :meth:`solve` method exposed on every
PyGROG operator (``SparseFFT``, ``MaskedFFT`` and their
:mod:`pygrog.gadgets` decorators), and the
:class:`pygrog.PolynomialPreconditioner` accelerator.

The pipeline mirrors :doc:`example01_basic_usage`:

1. Simulate non-Cartesian k-space from a BrainWeb phantom.
2. GROG-grid into Cartesian space (sparse and dense paths).
3. Solve the regularised least-squares problem

   .. math::

       \\hat x \\;=\\; \\arg\\min_x \\; \\| A x - b \\|_2^2
                                    + \\lambda^2 \\| x \\|_2^2

   using :func:`pygrog.cg`, :func:`pygrog.lsmr`, and a polynomial-
   preconditioned :func:`pygrog.cg`.
"""

import matplotlib.pyplot as plt
import numpy as np
from brainweb_dl import get_mri
from mrinufft import get_operator, initialize_2D_spiral
from mrinufft.density import voronoi

from pygrog import PolynomialPreconditioner
from pygrog.calib import GrogInterpolator
from pygrog.operator import MaskedFFT, SparseFFT

# sphinx_gallery_start_ignore
def _synthetic_smaps(shape, n_coils=4):
    ny, nx = shape
    yy, xx = np.mgrid[-1 : 1 : ny * 1j, -1 : 1 : nx * 1j]
    smaps = []
    for angle in np.linspace(0.0, 2.0 * np.pi, n_coils, endpoint=False):
        cx, cy = np.cos(angle), np.sin(angle)
        gauss = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * 0.45**2))
        phase = np.exp(1j * (cx * xx + cy * yy))
        smaps.append(gauss * phase)
    smaps = np.asarray(smaps, dtype=np.complex64)
    smaps /= np.sqrt((np.abs(smaps) ** 2).sum(0, keepdims=True)) + 1e-12
    return smaps
# sphinx_gallery_end_ignore


# %%
image = get_mri(0, "T1")
# sphinx_gallery_start_ignore
image = np.flip(image, axis=(0, 1, 2))[90].astype(np.float32)
image /= image.max() + 1e-8
shape = image.shape
n_coils = 16

samples = initialize_2D_spiral(Nc=48, Ns=600, nb_revolutions=10).astype(np.float32)
density = voronoi(samples)
smaps = _synthetic_smaps(shape, n_coils=n_coils)
# sphinx_gallery_end_ignore

# %%
nufft = get_operator("finufft")(
# sphinx_gallery_start_ignore
    samples=samples,
    shape=shape,
    n_coils=n_coils,
    smaps=smaps,
    density=density,
    squeeze_dims=True,
)
kspace_nc = nufft.op(image.astype(np.complex64))  # (n_coils, n_samples)
print(f"k-space shape: {kspace_nc.shape}")
# sphinx_gallery_end_ignore

# %%
# GROG calibration and gridding
# =============================
#
# Set up :class:`~pygrog.calib.GrogInterpolator` to grid the non-Cartesian
# k-space onto Cartesian samples. Prepare both a sparse path and a dense
# (gridded) path for comparison.

calib = nufft.adj_op(kspace_nc)  # quick low-res-ish phantom estimate
calib = calib.astype(np.complex64, copy=False)[None]  # (1, *shape)

grog = GrogInterpolator(
    shape=shape,
    coords=samples.reshape(48, 600, 2),
    kernel_width=2,
    oversamp=2.0,
    image_shape=shape,
)

calib_full = smaps * image.astype(
    np.complex64
)  # (n_coils, *shape) — ground-truth coil images


grog.calc_interp_table(calib_full, lamda=0.01, precision=1)

kspace_nc_shaped = kspace_nc.reshape(n_coils, 48, 600)

# Sparse path
sparse = grog.interpolate(kspace_nc_shaped, ret_image=False)
op_sp = SparseFFT(plan=grog.plan, smaps=smaps)
b_sp = sparse * np.asarray(grog.plan.pre_weights)
b_sp = b_sp.reshape(*b_sp.shape[:-1], *op_sp.natural_shape)

# Dense/grid path
kgrid, mplan = grog.interpolate(kspace_nc_shaped, grid=True)
op_m = MaskedFFT(plan=mplan, smaps=smaps)
b_m = kgrid


# %%
# Conjugate Gradient (CG) Solver
# ==============================
#
# Solve the regularized normal equations using CG. PyGROG operators
# automatically dispatch to CG when Toeplitz acceleration is available.

# %%
residuals_cg = []
# sphinx_gallery_start_ignore


def _cb_cg(_k, _x, rel):
    residuals_cg.append(rel)


residuals_lsmr = []


def _cb_lsmr(_k, _x, rel):
    residuals_lsmr.append(rel)


residuals_pcg = []


def _cb_pcg(_k, _x, rel):
    residuals_pcg.append(rel)
# sphinx_gallery_end_ignore


img_cg = op_m.solve(
    b_m,
    method="cg",
    max_iter=20,
    damp=1e-3,
    callback=_cb_cg,
)
print(f"CG: {len(residuals_cg)} iterations")

# %%
# LSMR Solver
# ===========
#
# Use LSMR (iterative solver for regularised least-squares) as an alternative
# to CG for comparison.

img_lsmr = op_m.solve(
    b_m,
    method="lsmr",
    max_iter=20,
    callback=_cb_lsmr,
)
print(f"LSMR: {len(residuals_lsmr)} iterations")

# %%
# Polynomial Preconditioned CG (PCG)
# ==================================
#
# :class:`pygrog.PolynomialPreconditioner` builds a polynomial approximation
# to the inverse of ``A^H A`` from Chebyshev polynomials. Each application
# requires ``degree`` matrix-vector products but typically accelerates CG
# convergence significantly.

pc = PolynomialPreconditioner(op_m, degree=3, n_power_iter=10)
print(f"Preconditioner spectrum estimate: {pc.spectrum}")
print(f"Polynomial coefficients: {pc.coeffs}")

img_pcg = op_m.solve(
    b_m,
    method="cg",
    max_iter=20,
    damp=1e-3,
    preconditioner=pc,
    callback=_cb_pcg,
)
print(f"PCG: {len(residuals_pcg)} iterations")

# %%
# Compare convergence
# ===================
fig, ax = plt.subplots(figsize=(6, 4))
# sphinx_gallery_start_ignore
ax.semilogy(residuals_cg, "-o", label="CG")
ax.semilogy(residuals_lsmr, "-s", label="LSMR")
ax.semilogy(residuals_pcg, "-^", label=f"PCG (deg={pc.degree})")
ax.set_xlabel("iteration")
ax.set_ylabel(r"$\|r_k\| / \|r_0\|$")
ax.set_title("Iterative solver convergence (MaskedFFT)")
ax.grid(True, which="both", alpha=0.3)
ax.legend()
plt.tight_layout()
plt.show()
# sphinx_gallery_end_ignore

# %%
# Visualise the reconstructions
# =============================

fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
# sphinx_gallery_start_ignore
ref = image / (image.max() + 1e-12)

for ax, im, title in zip(
    axes,
    [
        ref,
        np.abs(np.asarray(img_cg)),
        np.abs(np.asarray(img_lsmr)),
        np.abs(np.asarray(img_pcg)),
    ],
    ["reference", "CG", "LSMR", f"PCG (deg={pc.degree})"],
    strict=False,
):
    arr = im / (im.max() + 1e-12)
    ax.imshow(arr, cmap="gray", origin="upper")
    ax.set_title(title)
    ax.axis("off")
plt.tight_layout()
plt.show()
# sphinx_gallery_end_ignore

# %%
# The same `solve()` API works on the sparse path
# ===============================================
img_cg_sp = op_sp.solve(b_sp, method="cg", max_iter=20, damp=1e-3)
print(
    "SparseFFT solve image shape:",
    tuple(img_cg_sp.shape),
)
