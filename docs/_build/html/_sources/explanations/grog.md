# GROG: GRAPPA Operator Gridding

```{admonition} Summary
GROG (GRAPPA Operator Gridding) extends the GRAPPA linear-prediction model
to non-Cartesian k-space.  It uses fractional GRAPPA operators — computed
as matrix exponentials of the unit-shift operators — to map each
non-Cartesian source point to its Cartesian neighbours.
```

## From Cartesian GRAPPA to Non-Cartesian Gridding

Cartesian GRAPPA uses **integer-step** displacement operators $G_x$ and
$G_y$ that shift k-space by one grid unit along each dimension.  If the
source and target differ by $(n_x, n_y)$ integer steps, the operator is

$$
G_{n_x,n_y} = G_x^{n_x}\, G_y^{n_y}.
$$

GROG (Seiberlich et al., 2007) generalises this to *arbitrary fractional
shifts* by replacing integer powers with matrix exponentials:

$$
G_{\delta_x, \delta_y} = e^{\delta_x \log G_x}\, e^{\delta_y \log G_y},
$$

where $\delta_x, \delta_y \in \mathbb{R}$ are the fractional displacements
from the non-Cartesian source to the target Cartesian grid point.

## Algorithm Outline

### Step 1 — Kernel training (once per trajectory)

Unit-shift GRAPPA operators $G_x$ and $G_y$ are computed from the ACR by
solving:

$$
\underset{G_x}{\min}\; \left\| G_x \mathbf{s} - \mathbf{s}_{+1_x} \right\|_F^2
+ \lambda \|G_x\|_F^2,
$$

where $\mathbf{s}$ is the multi-coil source matrix and $\mathbf{s}_{+1_x}$
is the same matrix shifted by one step in $x$.  The matrix logarithm
$\log G_x$ is computed once via eigendecomposition and cached.

PyGROG implements this in {func}`pygrog.calib.KernelTable`.

### Step 2 — Fractional operator table

For every unique displacement $(\delta_x, \delta_y)$ encountered in the
trajectory (quantised to a user-specified decimal precision), a precomputed
fractional operator

$$
G(\delta_x, \delta_y) = e^{\delta_x \log G_x} \cdot e^{\delta_y \log G_y}
$$

is stored in a look-up table.  The look-up table is built once and
reused for every dataset acquired with the same trajectory.

### Step 3 — Gridding (per dataset)

For each non-Cartesian source point $\mathbf{k}_s$ and each Cartesian
target $\mathbf{k}_t$ in its neighbourhood, the gridded value is:

$$
\hat{y}(\mathbf{k}_t) \mathrel{+}=
  G(\mathbf{k}_t - \mathbf{k}_s)\, y(\mathbf{k}_s),
$$

where $y(\mathbf{k}_s) \in \mathbb{C}^L$ is the $L$-coil source vector.
The scatter-accumulate is executed in a single batched CUDA kernel in
PyGROG ({class}`pygrog.calib.GrogInterpolator`).

## Complexity and Accuracy

| Quantity | Cartesian GRAPPA | GROG |
|---|---|---|
| Training cost | $O(N_\text{acr}^2 \cdot L^3)$ | same |
| Gridding cost | $O(N_s \cdot k_w \cdot L^2)$ | same |
| Extra memory | kernel table (size $N_\text{steps}^d \times L^2$) | exp table (size $N_\delta \times L^2$) |
| Accuracy | exact (on Cartesian grid) | limited by quantisation precision |

The quantisation precision `precision` (decimal digits) trades accuracy
for table size. Typical values are 1–2 decimal digits.

:::{note}
GROG is a **non-iterative** gridding method.  It is faster than NUFFT-based
methods for large batches of radial or spiral data, but the output is still
a density-weighted approximation.  Iterative reconstruction (e.g. CG-SENSE)
on top of {class}`~pygrog.operator.SparseFFT` yields quantitatively
accurate images.
:::

## Comparison with the NUFFT

The classical non-uniform FFT (NUFFT) and GROG both solve the same
non-Cartesian sampling problem, but with different trade-offs:

| | NUFFT | GROG |
|---|---|---|
| Gridding kernel | fixed (Kaiser-Bessel) | data-driven (GRAPPA) |
| Coil handling | single operator | all coils simultaneously |
| Calibration data | none required | ACR required |
| GPU efficiency | excellent (cuFINUFFT) | excellent (custom scatter kernel) |
| Trajectory change | fast (only recompute plan) | requires kernel retraining |

PyGROG is designed to **complement** mri-nufft: use GROG for fast
online gridding when a multi-coil ACR is available, and fall back to a
NUFFT backend when calibration data are unavailable.

## References

- Seiberlich N, et al. *Non-Cartesian data reconstruction using GRAPPA
  operator gridding (GROG).* Magn Reson Med. 2007;58(6):1257-65.
- Seiberlich N, et al. *Improved radial GRAPPA calibration for real-time
  free-breathing cardiac imaging.* Magn Reson Med. 2011;65(2):492-505.
- Griswold MA, et al. *Generalized autocalibrating partially parallel
  acquisitions (GRAPPA).* Magn Reson Med. 2002;47(6):1202-10.
