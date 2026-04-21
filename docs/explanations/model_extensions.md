# Model Extensions

The basic GROG gridding model can be extended to account for physical
effects that are present in real MRI acquisitions.  PyGROG implements three
main extensions via *gadgets* that wrap a base
{class}`~pygrog.operator.SparseFFT` operator.

---

## Parallel Imaging (Multi-Coil)

The multi-coil acquisition model is:

$$
y_{i,\ell} = \int S_\ell(\mathbf{r})\, x(\mathbf{r})\,
             e^{-2\pi i \mathbf{r} \cdot \mathbf{k}_i}\, d\mathbf{r}
           + n_{i,\ell}, \quad \ell = 1, \ldots, L,
$$

or in operator form:

$$
\tilde{\mathbf{y}} =
\begin{bmatrix}
\mathcal{F}_\Omega S_1 \\
\vdots \\
\mathcal{F}_\Omega S_L
\end{bmatrix} x + \mathbf{n}
= \tilde{\mathcal{F}}_\Omega x + \mathbf{n}.
$$

{class}`~pygrog.operator.SparseFFT` supports coil sensitivity maps via the
`smaps` argument.  When `smaps` is provided, the **forward** direction
(image → k-space, i.e. the adjoint NUFFT direction) expands the image with
each coil map before applying the FFT:

$$
(\tilde{\mathcal{F}}_\Omega^* x)_\ell = \mathcal{F}_\Omega^* (S_\ell^* x),
$$

and the **adjoint** direction (k-space → image, i.e. the forward NUFFT
direction) performs coil-combination:

$$
\tilde{\mathcal{F}}_\Omega \tilde{y}
= \sum_{\ell=1}^L S_\ell^* \mathcal{F}_\Omega \tilde{y}_\ell.
$$

Coil sensitivity maps can be estimated from the ACR using
{func}`~pygrog.utils.nlinv_calib` (NLINV algorithm).

---

## Low-Rank Temporal Subspace

In dynamic MRI (cardiac, quantitative mapping, etc.) the image evolves over
time.  Instead of reconstructing each frame independently, the temporal
signal is constrained to a low-rank subspace spanned by $K \ll T$ basis
vectors $\{\phi_k\}_{k=1}^K$:

$$
x(\mathbf{r}, t) \approx \sum_{k=1}^K \alpha_k(\mathbf{r})\, \phi_k(t).
$$

The $K$ spatial coefficient maps $\{\alpha_k\}$ are the unknowns; the basis
$\Phi \in \mathbb{C}^{K \times T}$ is computed once from simulated or
measured signal dictionaries via truncated SVD.

The extended encoding operator maps coefficients to multi-frame k-space:

$$
y_t = \mathcal{F}_\Omega \left( \sum_{k=1}^K \phi_k(t)\, \alpha_k \right)
    = \sum_{k=1}^K \phi_k(t)\, \mathcal{F}_\Omega \alpha_k.
$$

PyGROG implements this with {class}`~pygrog.gadgets.SubspaceSparseFFT`,
which fuses the basis projection directly into the sparse FFT.  The
standalone {class}`~pygrog.gadgets.SubspaceProjection` class handles the
projection/expansion alone (without the FFT), which is useful for
post-processing or preconditioning.

---

## Off-Resonance Correction

B0 field inhomogeneities cause a spatially and temporally varying phase
during readout:

$$
y(t) = \int S(\mathbf{r})\, x(\mathbf{r})\,
       e^{i 2\pi \Delta f(\mathbf{r})\, t}\,
       e^{-2\pi i \mathbf{r} \cdot \mathbf{k}(t)}\, d\mathbf{r},
$$

where $\Delta f(\mathbf{r})$ is the B0 field map in Hz and $t$ is the
readout time of sample $\mathbf{k}(t)$.

The off-resonance exponential is approximated by a low-rank factorisation
(Sutton et al., 2003):

$$
e^{i 2\pi \Delta f(\mathbf{r})\, t} \approx
\sum_{\ell=1}^{L_\text{orc}} B_\ell(t)\, C_\ell(\mathbf{r}),
$$

where $B_\ell \in \mathbb{C}^{n_\text{samples}}$ and
$C_\ell \in \mathbb{C}^{n_x \times n_y}$ are the temporal and spatial
basis functions obtained by SVD of the phase modulation matrix.

The extended operator is:

$$
y(t) \approx \sum_{\ell=1}^{L_\text{orc}} B_\ell(t)\,
\mathcal{F}_\Omega [C_\ell(\mathbf{r})\, S(\mathbf{r})\, x(\mathbf{r})],
$$

which requires only $L_\text{orc}$ standard FFT evaluations.
PyGROG implements this in {class}`~pygrog.gadgets.OffResonanceCorrection`,
reusing the factorisation from `mri-nufft`.

:::{tip}
All three extensions can be combined: use a
{class}`~pygrog.gadgets.SubspaceSparseFFT` as the base operator for
{class}`~pygrog.gadgets.OffResonanceCorrection` to obtain a joint
subspace + off-resonance corrected operator.
:::

---

## References

- Griswold MA, et al. *GRAPPA.* Magn Reson Med. 2002;47(6):1202-10.
- Seiberlich N, et al. *GROG.* Magn Reson Med. 2007;58(6):1257-65.
- Uecker M, et al. *NLINV.* Magn Reson Med. 2008;60(3):674-82.
- Sutton BP, et al. *Fast, iterative image reconstruction for MRI in the
  presence of field inhomogeneities.* IEEE Trans Med Imaging. 2003;22(2):178-88.
- Liang ZP. *Spatiotemporal imaging with partially separable functions.*
  IEEE ISBI. 2007:988-91.
