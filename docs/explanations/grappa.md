# The GRAPPA Algorithm

```{admonition} Summary
GRAPPA (GeneRalized Autocalibrating Partial Parallel Acquisition) is a
parallel-imaging reconstruction algorithm that estimates missing k-space
lines by linear combination of acquired multi-coil data using kernels
trained from a calibration region.
```

## Background: Parallel Imaging in MRI

Modern MRI scanners use arrays of receive coils, each with a distinct
spatial sensitivity profile $S_\ell(\mathbf{r})$.  For $L$ coils the
measured signal is:

$$
y_{i,\ell} = \int_{\mathbb{R}^d} S_\ell(\mathbf{r})\, x(\mathbf{r})\,
             e^{-2\pi i \mathbf{r} \cdot \mathbf{k}_i}\, d\mathbf{r}
           + n_{i,\ell}
$$

where $\mathbf{k}_i$ is the $i$-th k-space sample location and $n_{i,\ell}$
is thermal noise.

In matrix form the acquisition is

$$
\mathbf{y} = \mathcal{F}_\Omega \mathbf{S}\, x + \mathbf{n},
$$

where $\mathcal{F}_\Omega$ is the Fourier operator restricted to the sampled
locations $\Omega$ and $\mathbf{S}$ stacks the coil sensitivity maps.

Undersampling $\Omega$ accelerates the scan but creates aliasing artefacts
in single-coil images.  Parallel imaging exploits coil-to-coil diversity to
undo the aliasing.

## The GRAPPA Linear Prediction Model

GRAPPA (Griswold et al., 2002) solves the aliasing problem entirely in
k-space.  The key observation is that any *unacquired* Cartesian k-space
point for coil $\ell$ can be expressed as a weighted sum of neighbouring
*acquired* points across all coils:

$$
\hat{y}(\mathbf{k}_0, \ell) =
  \sum_{\ell'=1}^{L} \sum_{j \in \mathcal{N}(\mathbf{k}_0)}
  w_{\ell,\ell'}(\mathbf{k}_0 - \mathbf{k}_j)\, y(\mathbf{k}_j, \ell'),
$$

where $\mathcal{N}(\mathbf{k}_0)$ is a local kernel neighbourhood and
$w_{\ell,\ell'}(\Delta\mathbf{k})$ are the kernel weights (one matrix per
displacement $\Delta\mathbf{k}$).

### Kernel training

Kernel weights are estimated from the **auto-calibration region (ACR)** —
a small, fully sampled central portion of k-space.  Within the ACR,
both source (neighbours) and target (central point) values are known, so the
weights are determined by least-squares:

$$
\underset{W}{\min}\;
\left\| \mathbf{A}\, W - \mathbf{B} \right\|_F^2 + \lambda \|W\|_F^2,
$$

where $\mathbf{A}$ is the source matrix (rows = ACR positions, columns =
neighbourhood values across all coils), $\mathbf{B}$ is the target matrix,
and $\lambda$ is Tikhonov regularisation.

PyGROG implements this in {func}`pygrog.calib.KernelTable`, which returns
a stack of kernel matrices indexed by discrete displacements.

## Practical Considerations

:::{tip}
Larger ACR regions yield more stable kernel estimates, at the cost of
extra scan time. Typical ACR widths are 24–32 lines in each phase-encode
direction.
:::

:::{note}
Tikhonov regularisation ($\lambda$) is critical for ill-conditioned kernels
(few coils or large kernel size).  PyGROG defaults to $\lambda = 0.01$.
:::

## References

- Griswold MA, et al. *Generalized autocalibrating partially parallel
  acquisitions (GRAPPA).* Magn Reason Med. 2002;47(6):1202-10.
- Uecker M, et al. *ESPIRiT — an eigenvalue approach to autocalibrating
  parallel MRI.* Magn Reason Med. 2014;71(3):990-1001.
