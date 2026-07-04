"""
SFFT — Space-Frequency Fourier Transform image subtraction.

Implements spatially varying kernel fitting in a single global least-squares
solve, following the approach of Hu et al. (2022, ApJ, 936, 157).

The model is::

    science(x,y) = Σ_α Σ_β c_{α,β} · [P_β(x,y) · R_α(x,y)] + Σ_γ d_γ · P_γ(x,y)

where R_α is the reference shifted by kernel offset α, P_β are polynomial
basis functions encoding spatial variation, and c_{α,β} / d_γ are scalar
coefficients solved for globally.

The normal equations are assembled via batched BLAS matmuls over polynomial
term pairs, avoiding materializing the full design matrix.

A soft kernel-sum constraint enforces that Σ_α a_α(x,y) = f(x,y) where f is
a low-order polynomial modelling smooth flux-scale variation across the image.
"""

import numpy as np
from dataclasses import dataclass
from scipy import linalg as scipy_linalg
from typing import Tuple


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class SFFTResult:
    """Result of SFFT image subtraction."""

    diff: np.ndarray
    """Difference image (science - model)."""

    model: np.ndarray
    """Convolved reference + background model."""

    kernel_coeffs: np.ndarray
    """Kernel coefficients, shape (n_kernel, n_kpoly)."""

    bg_coeffs: np.ndarray
    """Background coefficients, shape (n_bgpoly,)."""

    kernel_shape: Tuple[int, int]
    """Kernel support size (ky, kx)."""

    kernel_poly_order: int
    """Polynomial order for kernel spatial variation."""

    bg_poly_order: int
    """Polynomial order for differential background."""

    flux_poly_order: int
    """Polynomial order for the kernel-sum (flux scale) constraint."""

    flux_poly_coeffs: np.ndarray
    """Fitted flux-scale polynomial coefficients."""

    n_iter: int
    """Number of solver passes performed (sigma-clipping iterations plus
    the template-noise reweighting pass, if any)."""

    rms: float
    """Final RMS of residuals over good (unmasked, unclipped) pixels."""

    n_good: int
    """Number of good (unmasked, unclipped) pixels in final iteration."""

    dmask: np.ndarray = None
    """Boolean mask (True = unreliable) of difference pixels whose model is
    built from missing template data: image edges and pixels within the
    kernel footprint of template defects (NaN/Inf or ``template_mask``)."""


# ---------------------------------------------------------------------------
# Polynomial helpers
# ---------------------------------------------------------------------------


def _poly_terms_2d(x, y, order):
    """Triangular 2-D polynomial basis evaluated on coordinate arrays.

    Term ordering: (0,0), (1,0), (0,1), (2,0), (1,1), (0,2), ...

    Parameters
    ----------
    x, y : ndarray, same shape (any)
    order : int >= 0

    Returns
    -------
    terms : ndarray, shape (n_terms, *x.shape)
    """
    terms = []
    for total in range(order + 1):
        for px in range(total + 1):
            py = total - px
            terms.append(x**px * y**py)
    return np.array(terms, dtype=np.float64)


def _n_poly(order):
    """Number of terms in a 2-D triangular polynomial of given order."""
    return (order + 1) * (order + 2) // 2


def _norm_coords(ny, nx):
    """Pixel coordinate arrays normalized to [-1, 1]."""
    yy, xx = np.indices((ny, nx), dtype=np.float64)
    cx, cy = 0.5 * (nx - 1), 0.5 * (ny - 1)
    sx = max(1.0, cx)
    sy = max(1.0, cy)
    return (xx - cx) / sx, (yy - cy) / sy


def _norm_xy(x, y, image_shape):
    """Normalize pixel coordinate value(s) to [-1, 1], matching :func:`_norm_coords`."""
    ny, nx = image_shape[:2]
    xn = (np.asarray(x, dtype=np.float64) - 0.5 * (nx - 1)) / max(1.0, 0.5 * (nx - 1))
    yn = (np.asarray(y, dtype=np.float64) - 0.5 * (ny - 1)) / max(1.0, 0.5 * (ny - 1))
    return xn, yn


# ---------------------------------------------------------------------------
# Kernel offset helpers
# ---------------------------------------------------------------------------


def _kernel_offsets(ky, kx):
    """List of (dy, dx) offsets for a kernel of shape (ky, kx). Both must be odd."""
    if ky % 2 == 0 or kx % 2 == 0:
        raise ValueError("kernel dimensions must be odd, got (%d, %d)" % (ky, kx))
    hy, hx = ky // 2, kx // 2
    return [(dy, dx) for dy in range(-hy, hy + 1) for dx in range(-hx, hx + 1)]


def _shift_image(img, dy, dx):
    """Shift 2-D array by integer (dy, dx), zero-filling exposed edges."""
    ny, nx = img.shape
    out = np.zeros_like(img)
    sy0, sy1 = max(0, -dy), min(ny, ny - dy)
    dy0 = sy0 + dy
    sx0, sx1 = max(0, -dx), min(nx, nx - dx)
    dx0 = sx0 + dx
    if sy1 > sy0 and sx1 > sx0:
        out[dy0 : dy0 + (sy1 - sy0), dx0 : dx0 + (sx1 - sx0)] = img[sy0:sy1, sx0:sx1]
    return out


def _shift_band(img, dy, dx, r0, r1, out):
    """Rows ``r0:r1`` of ``_shift_image(img, dy, dx)``, written into ``out``.

    ``out`` must be a 2-D array of shape (r1 - r0, nx). Used to assemble
    the normal equations in row bands without materializing full shifted
    copies of the image.
    """
    ny, nx = img.shape
    out[:] = 0.0
    # Valid output rows of the full shift are [max(0, dy), min(ny, ny + dy))
    y0 = max(r0, dy)
    y1 = min(r1, ny + dy)
    sx0, sx1 = max(0, -dx), min(nx, nx - dx)
    dx0 = sx0 + dx
    if y1 > y0 and sx1 > sx0:
        out[y0 - r0 : y1 - r0, dx0 : dx0 + (sx1 - sx0)] = img[y0 - dy : y1 - dy, sx0:sx1]
    return out


# ---------------------------------------------------------------------------
# Core SFFT solve
# ---------------------------------------------------------------------------


# Target size (in elements) for the per-band (n_kernel, band_pix) work arrays
# used by _assemble_normal_equations: 2**25 float64 elements = 256 MB.
_ASSEMBLE_BAND_ELEMENTS = 2**25

# Downdate the accumulated normal equations by the clipped pixels'
# contributions instead of reassembling from scratch after each
# sigma-clipping iteration. Exact up to floating-point roundoff; the flag
# exists for testing and debugging.
_USE_INCREMENTAL_UPDATES = True


def _accumulate_normal_blocks(H, g, srefs, w_c, s_c, pk_c, pbg_c, n_kernel_params):
    """Accumulate a pixel subset's contributions into H (upper blocks) and g.

    Every term is linear in the pixel weight ``w_c``, so calling this with
    negated weights exactly subtracts (downdates) previously accumulated
    contributions.

    Only the upper triangle of the kernel-kernel block structure (b1 <= b2
    polynomial pairs) and the kernel→background cross rows are filled;
    use :func:`_mirror_normal_matrix` to complete the symmetric matrix.

    Parameters
    ----------
    H : (n_total, n_total), modified in place
    g : (n_total,), modified in place
    srefs : (n_kernel, n_pix) shifted reference values at the subset pixels
    w_c : (n_pix,) pixel weights (may be negative for downdating)
    s_c : (n_pix,) science values
    pk_c : (n_kpoly, n_pix) kernel polynomial terms at the subset pixels
    pbg_c : (n_bgpoly, n_pix) background polynomial terms at the subset pixels
    n_kernel_params : number of kernel parameters (offset × poly terms)
    """
    n_kpoly = pk_c.shape[0]

    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        # === Background-background block ===
        w_poly_bg = pbg_c * w_c[np.newaxis, :]  # (n_bgpoly, n_pix)
        H[n_kernel_params:, n_kernel_params:] += w_poly_bg @ pbg_c.T
        g[n_kernel_params:] += w_poly_bg @ s_c

        # === Kernel RHS and kernel-background cross-block ===
        # g[α*n_kpoly + β] = Σ_pix w * P_β * R_α * S
        # H[α*n_kpoly+β, n_kernel_params+γ] = Σ_pix w * P_β * R_α * Q_γ
        ws_c = w_c * s_c
        for b in range(n_kpoly):
            g[b:n_kernel_params:n_kpoly] += srefs @ (ws_c * pk_c[b])
            wPb_Q = pbg_c * (w_c * pk_c[b])[np.newaxis, :]  # (n_bgpoly, n_pix)
            # Rows α*n_kpoly+b for all α form the strided slice below
            H[b:n_kernel_params:n_kpoly, n_kernel_params:] += srefs @ wPb_Q.T

        # === Kernel-kernel block (the main bottleneck) ===
        # H[α1*n_kpoly+β1, α2*n_kpoly+β2] = Σ_pix w * P_β1 * R_α1 * P_β2 * R_α2
        # For fixed (β1, β2):
        #   M[α1, α2] = (shifted_refs * (w*P_β1)) @ (shifted_refs * P_β2).T
        # This is n_kpoly*(n_kpoly+1)/2 matmuls of size (n_kernel × n_pix) @ (n_pix × n_kernel);
        # only the b1 <= b2 blocks are accumulated.
        for b1 in range(n_kpoly):
            wPb1_refs = srefs * (w_c * pk_c[b1])[np.newaxis, :]  # (n_kernel, n_pix)
            for b2 in range(b1, n_kpoly):
                M = wPb1_refs @ (srefs * pk_c[b2][np.newaxis, :]).T
                H[b1:n_kernel_params:n_kpoly, b2:n_kernel_params:n_kpoly] += M


def _mirror_normal_matrix(H, n_kernel_params, n_kpoly):
    """Mirror the accumulated upper blocks of H into the lower triangle."""
    for b1 in range(n_kpoly):
        for b2 in range(b1 + 1, n_kpoly):
            H[b2:n_kernel_params:n_kpoly, b1:n_kernel_params:n_kpoly] = H[
                b1:n_kernel_params:n_kpoly, b2:n_kernel_params:n_kpoly
            ].T
    H[n_kernel_params:, :n_kernel_params] = H[:n_kernel_params, n_kernel_params:].T
    return H


def _assemble_normal_equations(
    reference, science, weight, kernel_shape, kernel_poly_order, bg_poly_order, x_norm, y_norm
):
    """Assemble normal equations H·θ = g via polynomial-pair batching.

    Instead of iterating over O(n_kernel²) offset pairs with small matmuls,
    iterates over O(n_kpoly²) polynomial term pairs with large BLAS matmuls.
    For typical parameters (7×7 kernel, poly=2), this is ~60× fewer iterations
    with much better BLAS utilization (49×49 matmuls vs 6×6).

    All contributions are plain sums over pixels, so the image is processed
    in bands of rows: peak memory is bounded by a few (n_kernel, band_pix)
    work arrays (~256 MB each) instead of the full (n_kernel, npix) set of
    shifted references, which would reach tens of GB for 4k images.

    Parameters
    ----------
    reference : (ny, nx)
    science : (ny, nx)
    weight : (ny, nx)
    kernel_shape, kernel_poly_order, bg_poly_order : model specification
    x_norm, y_norm : (ny, nx) normalized coordinates

    Returns
    -------
    H : (n_total, n_total) normal matrix, upper-block form — pass through
        :func:`_mirror_normal_matrix` before solving
    g : (n_total,) right-hand side
    """
    ky, kx = kernel_shape
    offsets = _kernel_offsets(ky, kx)
    n_kernel_pixels = ky * kx
    n_kpoly = _n_poly(kernel_poly_order)
    n_bgpoly = _n_poly(bg_poly_order)
    n_kernel_params = n_kernel_pixels * n_kpoly
    n_total = n_kernel_params + n_bgpoly

    ny, nx = reference.shape
    npix = ny * nx

    poly_k = _poly_terms_2d(x_norm, y_norm, kernel_poly_order)  # (n_kpoly, ny, nx)
    poly_bg = _poly_terms_2d(x_norm, y_norm, bg_poly_order)  # (n_bgpoly, ny, nx)

    # Flatten everything for matrix ops
    poly_k_flat = poly_k.reshape(n_kpoly, npix)  # (n_kpoly, npix)
    poly_bg_flat = poly_bg.reshape(n_bgpoly, npix)  # (n_bgpoly, npix)
    w_flat = weight.ravel()  # (npix,)
    s_flat = science.ravel()  # (npix,)

    H = np.zeros((n_total, n_total), dtype=np.float64)
    g = np.zeros(n_total, dtype=np.float64)

    band_rows = max(1, min(ny, _ASSEMBLE_BAND_ELEMENTS // (n_kernel_pixels * nx)))
    shifted_refs = np.empty((n_kernel_pixels, band_rows * nx), dtype=np.float64)

    for r0 in range(0, ny, band_rows):
        r1 = min(ny, r0 + band_rows)
        bpix = (r1 - r0) * nx
        sl = slice(r0 * nx, r1 * nx)

        # Shifted references restricted to this band: (n_kernel, bpix)
        srefs = shifted_refs[:, :bpix]
        for a, (dy, dx) in enumerate(offsets):
            _shift_band(reference, dy, dx, r0, r1, srefs[a].reshape(r1 - r0, nx))

        _accumulate_normal_blocks(
            H, g, srefs, w_flat[sl], s_flat[sl],
            poly_k_flat[:, sl], poly_bg_flat[:, sl], n_kernel_params,
        )

    return H, g


def _downdate_normal_equations(
    H, g, reference, science, w_pix, ys, xs,
    kernel_shape, kernel_poly_order, bg_poly_order, x_norm, y_norm,
):
    """Subtract individual pixels' contributions from accumulated normal equations.

    Since every term of H and g is linear in the pixel weight, removing
    pixels (sigma-clipping) can be done exactly by accumulating their
    contributions with negated weights — far cheaper than reassembling
    from the full image when only a small fraction of pixels is clipped.

    ``H`` must be in the upper-block (pre-mirror) form produced by
    :func:`_assemble_normal_equations`.

    Parameters
    ----------
    H, g : accumulated normal equations, modified in place
    reference, science : (ny, nx) images
    w_pix : (n_pix,) weights the pixels contributed with
    ys, xs : (n_pix,) pixel coordinates
    kernel_shape, kernel_poly_order, bg_poly_order : model specification
    x_norm, y_norm : (ny, nx) normalized coordinates
    """
    ky, kx = kernel_shape
    hy, hx = ky // 2, kx // 2
    offsets = _kernel_offsets(ky, kx)
    n_kpoly = _n_poly(kernel_poly_order)
    n_kernel_params = ky * kx * n_kpoly

    # Shifted reference values R_α(p) = ref(p - offset_α) at scattered pixels,
    # gathered from a zero-padded copy so out-of-image offsets read 0 exactly
    # like _shift_image does
    ref_pad = np.pad(reference, ((hy, hy), (hx, hx)))
    dys = np.array([o[0] for o in offsets])
    dxs = np.array([o[1] for o in offsets])
    srefs = ref_pad[
        ys[np.newaxis, :] - dys[:, np.newaxis] + hy,
        xs[np.newaxis, :] - dxs[:, np.newaxis] + hx,
    ]  # (n_kernel, n_pix)

    pk_c = _poly_terms_2d(x_norm[ys, xs], y_norm[ys, xs], kernel_poly_order)
    pbg_c = _poly_terms_2d(x_norm[ys, xs], y_norm[ys, xs], bg_poly_order)

    _accumulate_normal_blocks(
        H, g, srefs, -w_pix, science[ys, xs], pk_c, pbg_c, n_kernel_params
    )


def _build_kernel_sum_constraint(kernel_shape, kernel_poly_order, flux_poly_order):
    """Build constraint matrix for kernel sum = flux polynomial.

    The kernel-sum at position (x,y) is:
        Σ_α a_α(x,y) = Σ_α Σ_β c_{α,β} · P_β(x,y)
                      = Σ_β [Σ_α c_{α,β}] · P_β(x,y)

    We want this to equal a flux polynomial:
        f(x,y) = Σ_γ f_γ · Q_γ(x,y)

    where Q_γ are polynomial terms up to flux_poly_order.

    If flux_poly_order <= kernel_poly_order, Q_γ ⊂ P_β, so the constraint is:
        Σ_α c_{α,β} = f_β   for β ≤ flux_poly_order terms
        Σ_α c_{α,β} = 0     for higher-order β terms

    This gives n_kpoly linear constraints on the kernel coefficients.

    Returns
    -------
    C : (n_constraints, n_kernel_params) or None
        Homogeneous constraint rows (C·θ_kernel = 0), or None if nothing
        needs constraining.
    n_flux_free : number of free flux polynomial coefficients
    """
    n_kernel_pixels = kernel_shape[0] * kernel_shape[1]
    n_kpoly = _n_poly(kernel_poly_order)
    n_flux = _n_poly(flux_poly_order)

    # We constrain: for each kernel poly term β with order > flux_poly_order,
    # the sum Σ_α c_{α,β} = 0.
    # For terms β within flux_poly_order, we don't constrain (those define
    # the free flux scale polynomial).

    # Count terms with total degree > flux_poly_order
    n_constrained = n_kpoly - n_flux
    if n_constrained <= 0:
        return None, n_flux

    # Build constraint rows
    # Parameter layout: [c_{0,0}, c_{0,1}, ..., c_{0,n_kpoly-1},
    #                    c_{1,0}, ..., c_{n_kernel-1, n_kpoly-1},
    #                    d_0, ..., d_{n_bgpoly-1}]
    # Total kernel params = n_kernel_pixels * n_kpoly
    # For poly term β: params at indices [α * n_kpoly + β for α in range(n_kernel_pixels)]

    n_total_kernel = n_kernel_pixels * n_kpoly
    C_rows = []

    # Identify which poly terms have total degree > flux_poly_order
    term_idx = 0
    for total in range(kernel_poly_order + 1):
        for px in range(total + 1):
            if total > flux_poly_order:
                # This term β should have Σ_α c_{α,β} = 0
                row = np.zeros(n_total_kernel, dtype=np.float64)
                for alpha in range(n_kernel_pixels):
                    row[alpha * n_kpoly + term_idx] = 1.0
                C_rows.append(row)
            term_idx += 1

    C = np.array(C_rows, dtype=np.float64)
    return C, n_flux


def _reconstruct_model(
    theta, reference, kernel_shape, kernel_poly_order, bg_poly_order, x_norm, y_norm
):
    """Reconstruct the model image from solved coefficients.

    model(x,y) = Σ_α [Σ_β c_{α,β} · P_β(x,y)] · R_α(x,y) + Σ_γ d_γ · Q_γ(x,y)

    Parameters
    ----------
    theta : (n_total,) solved coefficients
    reference : (ny, nx)
    kernel_shape, kernel_poly_order, bg_poly_order : model specification
    x_norm, y_norm : (ny, nx) normalized coordinates

    Returns
    -------
    model : (ny, nx)
    """
    ky, kx = kernel_shape
    offsets = _kernel_offsets(ky, kx)
    n_kpoly = _n_poly(kernel_poly_order)
    n_bgpoly = _n_poly(bg_poly_order)
    n_kernel_pixels = ky * kx

    poly_k = _poly_terms_2d(x_norm, y_norm, kernel_poly_order)  # (n_kpoly, ny, nx)
    poly_bg = _poly_terms_2d(x_norm, y_norm, bg_poly_order)  # (n_bgpoly, ny, nx)

    kernel_coeffs = theta[: n_kernel_pixels * n_kpoly].reshape(n_kernel_pixels, n_kpoly)
    bg_coeffs = theta[n_kernel_pixels * n_kpoly :]

    ny, nx = reference.shape
    npix = ny * nx

    # model = Σ_α a_α(x,y) · R_α(x,y) where a_α = Σ_β c_{α,β} · P_β
    # Accumulate one offset at a time to avoid materializing the full
    # (n_kernel, npix) arrays of coefficient maps and shifted references.
    poly_k_flat = poly_k.reshape(n_kpoly, npix)
    model_flat = np.zeros(npix, dtype=np.float64)
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        for alpha, (dy, dx) in enumerate(offsets):
            a_map = kernel_coeffs[alpha] @ poly_k_flat  # (npix,)
            model_flat += a_map * _shift_image(reference, dy, dx).ravel()

    model = model_flat.reshape(ny, nx)

    # Background contribution (vectorized)
    model += np.tensordot(bg_coeffs, poly_bg, axes=(0, 0))

    return model


def _propagate_template_variance(kernel_coeffs, kernel_shape, kernel_poly_order, var_tmpl):
    """Propagate template variance through the spatially varying kernel.

    var_conv(x,y) = Σ_α a_α²(x,y) · var_tmpl(x - offset_α)

    where a_α(x,y) are the kernel coefficient maps. This is the template
    noise contribution to the difference image (and to proper fit weights).

    Parameters
    ----------
    kernel_coeffs : (n_kernel, n_kpoly) solved kernel coefficients
    kernel_shape : (ky, kx)
    kernel_poly_order : polynomial order of kernel spatial variation
    var_tmpl : (ny, nx) template variance map

    Returns
    -------
    var_conv : (ny, nx) kernel-convolved template variance
    """
    ny, nx = var_tmpl.shape
    npix = ny * nx
    offsets = _kernel_offsets(*kernel_shape)
    n_kpoly = _n_poly(kernel_poly_order)

    x_norm, y_norm = _norm_coords(ny, nx)
    poly_k_flat = _poly_terms_2d(x_norm, y_norm, kernel_poly_order).reshape(n_kpoly, npix)

    # Accumulate one offset at a time, computing each coefficient map
    # a_α(x,y) = kernel_coeffs[α] @ poly_k on the fly to avoid
    # materializing the full (n_kernel, npix) array
    var_conv = np.zeros(npix, dtype=np.float64)
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        for alpha, (dy, dx) in enumerate(offsets):
            a_map = kernel_coeffs[alpha] @ poly_k_flat  # (npix,)
            shifted_var = _shift_image(var_tmpl, dy, dx).ravel()
            var_conv += a_map**2 * shifted_var

    return var_conv.reshape(ny, nx)


def _extract_flux_poly(theta, kernel_shape, kernel_poly_order, flux_poly_order):
    """Extract the fitted flux-scale polynomial from kernel coefficients.

    flux(x,y) = Σ_α a_α(x,y) = Σ_β [Σ_α c_{α,β}] · P_β(x,y)

    Only terms with total degree ≤ flux_poly_order contribute (higher
    terms are constrained to zero by the kernel-sum constraint).

    Returns
    -------
    flux_coeffs : (n_flux,) polynomial coefficients for the flux scale
    """
    n_kernel_pixels = kernel_shape[0] * kernel_shape[1]
    n_kpoly = _n_poly(kernel_poly_order)
    n_flux = _n_poly(flux_poly_order)

    kernel_coeffs = theta[: n_kernel_pixels * n_kpoly].reshape(n_kernel_pixels, n_kpoly)

    # Sum over all kernel pixels for each poly term
    sum_per_poly = kernel_coeffs.sum(axis=0)  # (n_kpoly,)

    # Return only the flux_poly_order terms
    return sum_per_poly[:n_flux]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def solve(
    image,
    template,
    mask=None,
    template_mask=None,
    err=None,
    template_err=None,
    kernel_shape=(7, 7),
    kernel_poly_order=2,
    bg_poly_order=2,
    flux_poly_order=1,
    flux_penalty=1e3,
    ridge=1e-6,
    sigma_clip=3.0,
    max_iter=5,
    verbose=False,
):
    """SFFT image subtraction with spatially varying kernel.

    Solves for a spatially varying convolution kernel and differential
    background in a single global least-squares problem. The kernel
    at each pixel is modelled as a delta-function basis with polynomial
    spatial variation. A soft constraint enforces that the kernel sum
    varies smoothly as a low-order polynomial (modelling flux-scale
    differences between science and template).

    Iterative sigma-clipping rejects outlier pixels (transients, cosmic
    rays, artifacts) that would otherwise bias the kernel solution.

    Since the template acts as the basis of the model, any template defect
    (NaN/Inf or ``template_mask``) corrupts the model within a full kernel
    footprint around it. Such pixels are excluded from the fit, and flagged
    in the ``dmask`` attribute of the result — the difference image values
    there (and in the kernel half-width band along image edges) are
    unreliable and should not be searched for transients.

    Parameters
    ----------
    image : numpy.ndarray
        Science image as a 2-D NumPy array.
    template : numpy.ndarray
        Template/reference image, same shape, aligned to science.
    mask : numpy.ndarray, optional
        Boolean mask where True = bad pixels to exclude from the fit
        (science-side defects, or regions to ignore). Unlike template
        defects, these only affect their own pixel.
    template_mask : numpy.ndarray, optional
        Mask of template defects (True = bad). Grown by the kernel footprint,
        since the model at every pixel within a kernel half-width of a
        template defect depends on the missing data. Pass template-side
        defects here rather than merging them into ``mask``.
    err : numpy.ndarray, optional
        Per-pixel error (standard deviation) map for inverse-variance
        weighting. If None, uniform weights are used. A constant error map
        (or True) is equivalent to uniform weighting and does not change
        the solution (unless ``template_err`` is also given).
    template_err : numpy.ndarray, optional
        Per-pixel error map of the template. If given, one reweighting pass
        is performed: after an initial solve, the template variance is
        propagated through the fitted kernel and the weights are rebuilt as
        ``1 / (err² + Σ_α a_α² · template_err²)``, followed by a re-solve.
        This properly de-weights pixels dominated by template noise and
        matters when the template is not much deeper than the science image.
        Costs one extra solver pass.
    kernel_shape : tuple of int, optional
        (ky, kx) size of the convolution kernel, must be odd. Default (7, 7).
        Larger kernels handle bigger PSF differences but are slower.
    kernel_poly_order : int, optional
        Polynomial order for spatial variation of each kernel coefficient.
        Default 2 (quadratic). Higher orders capture more complex PSF variation
        but need more pixels.
    bg_poly_order : int, optional
        Polynomial order for the differential background model. Default 2.
    flux_poly_order : int, optional
        Polynomial order for the kernel-sum constraint (flux scale variation).
        Default 1 (linear gradient). Set to 0 for constant flux scale.
    flux_penalty : float, optional
        Dimensionless penalty weight for the kernel-sum constraint, relative
        to the median diagonal of the normal matrix (so its effect does not
        depend on image size or flux units). Default 1e3, which enforces the
        constraint nearly strictly. Set to 0 to disable the constraint
        entirely.
    ridge : float, optional
        Dimensionless Tikhonov regularization, relative to the median
        diagonal of the normal matrix. Default 1e-6.
    sigma_clip : float, optional
        Sigma threshold for iterative outlier rejection. Default 3.0. Set to
        None or 0 to disable clipping.
    max_iter : int, optional
        Maximum number of sigma-clipping iterations. Default 5. Must be at
        least 1. The reweighting pass triggered by ``template_err`` is
        performed in addition to these.
    verbose : bool or callable, optional
        If True, print progress. If callable, use as log function.

    Returns
    -------
    SFFTResult
        Result object with difference image and all fit metadata. Pixels
        flagged in its ``dmask`` attribute are unreliable in the difference.
    """

    log = (verbose if callable(verbose) else print) if verbose else lambda *args, **kwargs: None

    # --- Input validation and preparation ---
    sci = np.asarray(image, dtype=np.float64)
    ref = np.asarray(template, dtype=np.float64)

    if sci.shape != ref.shape:
        raise ValueError(
            "science and template must have the same shape, got %s vs %s" % (sci.shape, ref.shape)
        )
    if sci.ndim != 2:
        raise ValueError("images must be 2-D arrays")

    ny, nx = sci.shape
    ky, kx = kernel_shape

    if ky % 2 == 0 or kx % 2 == 0:
        raise ValueError("kernel_shape must be odd, got (%d, %d)" % (ky, kx))

    if flux_poly_order > kernel_poly_order:
        raise ValueError(
            "flux_poly_order (%d) must be <= kernel_poly_order (%d)"
            % (flux_poly_order, kernel_poly_order)
        )

    if max_iter < 1:
        raise ValueError("max_iter must be at least 1, got %d" % max_iter)

    # Replace NaN/Inf with 0 to prevent propagation through matmuls.
    # These pixels are masked via the weight map, so the values don't matter.
    nan_sci = ~np.isfinite(sci)
    nan_ref = ~np.isfinite(ref)
    if np.any(nan_sci) or np.any(nan_ref):
        sci = sci.copy()
        ref = ref.copy()
        sci[nan_sci] = 0.0
        ref[nan_ref] = 0.0

    n_kernel_pixels = ky * kx
    n_kpoly = _n_poly(kernel_poly_order)
    n_bgpoly = _n_poly(bg_poly_order)
    n_kernel_params = n_kernel_pixels * n_kpoly
    n_total = n_kernel_params + n_bgpoly

    n_fpoly = _n_poly(flux_poly_order)
    log(
        "SFFT: image %dx%d, kernel %dx%d, kernel_poly=%d (%d terms), "
        "bg_poly=%d (%d terms), flux_poly=%d (%d terms)"
        % (
            nx,
            ny,
            kx,
            ky,
            kernel_poly_order,
            n_kpoly,
            bg_poly_order,
            n_bgpoly,
            flux_poly_order,
            n_fpoly,
        )
    )
    log(
        "SFFT: %d kernel + %d background = %d total parameters"
        % (n_kernel_params, n_bgpoly, n_total)
    )

    # --- Build good-pixel mask ---
    # `good` tracks which pixels participate in the fit; it only ever shrinks
    # (masking, clipping). Float weights are derived from it separately so
    # that reweighting cannot resurrect clipped pixels.
    good = np.ones((ny, nx), dtype=bool)

    if mask is not None:
        good[np.asarray(mask, dtype=bool)] = False

    # Science NaN/Inf pixels only corrupt their own residual
    if np.any(nan_sci):
        good[nan_sci] = False

    # Template defects (masked or non-finite) corrupt the model everywhere
    # the kernel footprint overlaps them: the shifted template values enter
    # the design matrix of every pixel within a kernel half-width. Grow the
    # defect mask by the kernel footprint before excluding those pixels.
    bad_tmpl = nan_ref.copy()
    if template_mask is not None:
        bad_tmpl |= np.asarray(template_mask, dtype=bool)

    dmask = np.zeros((ny, nx), dtype=bool)
    if np.any(bad_tmpl):
        for dy, dx in _kernel_offsets(ky, kx):
            dmask |= _shift_image(bad_tmpl, dy, dx)
        n_grown = int(np.sum(dmask & ~bad_tmpl))
        if n_grown:
            log(
                "SFFT: template defects grown by kernel footprint: "
                "%d -> %d pixels" % (int(np.sum(bad_tmpl)), int(np.sum(dmask)))
            )
        good[dmask] = False

    # Edge pixels are likewise affected by zero-filled kernel shifts
    hy, hx = ky // 2, kx // 2
    if hy > 0:
        good[:hy, :] = False
        good[-hy:, :] = False
        dmask[:hy, :] = True
        dmask[-hy:, :] = True
    if hx > 0:
        good[:, :hx] = False
        good[:, -hx:] = False
        dmask[:, :hx] = True
        dmask[:, -hx:] = True

    # --- Inverse-variance base weights ---
    def _make_weight(var_total):
        """Weight map 1/var with invalid entries zeroed; None = uniform."""
        w = np.ones((ny, nx), dtype=np.float64)
        if var_total is not None:
            valid = (var_total > 0) & np.isfinite(var_total)
            w[valid] = 1.0 / var_total[valid]
            w[~valid] = 0.0
        return w

    if err is not None and err is not True:
        err_map = np.asarray(err, dtype=np.float64)
        var_sci = err_map**2
    else:
        var_sci = None
        if err is True:
            # A constant error map rescales all weights uniformly and does
            # not change the solution, so uniform weighting is used instead.
            log("SFFT: err=True has no effect; pass a per-pixel error map for weighting")

    if template_err is not None:
        template_err = np.asarray(template_err, dtype=np.float64)
        if template_err.shape != (ny, nx):
            raise ValueError("template_err must have the same shape as the images")

    base_weight = _make_weight(var_sci)

    # --- Normalized coordinates ---
    x_norm, y_norm = _norm_coords(ny, nx)

    # --- Build kernel-sum constraint ---
    C_kernel, _ = _build_kernel_sum_constraint(
        kernel_shape, kernel_poly_order, flux_poly_order
    )

    if C_kernel is not None and flux_penalty > 0:
        log(
            "SFFT: kernel-sum constraint: %d equations, flux_poly_order=%d, "
            "penalty=%.1e" % (C_kernel.shape[0], flux_poly_order, flux_penalty)
        )
        # The constraint only involves kernel parameters; its normal-equation
        # contribution (CᵀC) is constant across iterations
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            CtC_kernel = C_kernel.T @ C_kernel  # (n_kernel_params, n_kernel_params)
    else:
        CtC_kernel = None

    # --- Iterative sigma-clipping solve ---
    do_clip = sigma_clip is not None and sigma_clip > 0 and max_iter > 1
    # One extra solver pass to rebuild the weights with template noise
    # propagated through the fitted kernel (which needs a first-pass solve)
    reweight_pending = template_err is not None
    total_iters = max_iter + (1 if reweight_pending else 0)

    n_iter_done = 0
    final_rms = np.nan
    # Accumulated data-term normal equations (upper-block form). Kept pristine
    # (no penalty/ridge) so clipped pixels can be downdated incrementally;
    # set to None whenever the weights change globally (reweighting).
    H_data = None
    g_data = None

    for iteration in range(total_iters):
        n_iter_done = iteration + 1
        weight = np.where(good, base_weight, 0.0)
        n_good = int(np.sum(weight > 0))

        if n_good < n_total + 10:
            raise RuntimeError("Too few good pixels (%d) for %d parameters" % (n_good, n_total))

        if H_data is None:
            # Assemble normal equations (processed in row bands to bound memory)
            log("SFFT: assembling normal equations (%d parameters)" % n_total)
            H_data, g_data = _assemble_normal_equations(
                ref, sci, weight, kernel_shape, kernel_poly_order, bg_poly_order, x_norm, y_norm
            )
        else:
            log("SFFT: reusing normal equations downdated by clipped pixels")

        H = _mirror_normal_matrix(H_data.copy(), n_kernel_params, n_kpoly)
        g = g_data.copy()

        # Typical magnitude of the data term, used to make the penalty and
        # ridge scale-free (independent of image size, flux units, weights)
        h_diag = np.diagonal(H)[:n_kernel_params]
        h_pos = h_diag[h_diag > 0]
        h_scale = np.median(h_pos) if len(h_pos) else 1.0

        # Add kernel-sum penalty to normal equations
        if CtC_kernel is not None:
            H[:n_kernel_params, :n_kernel_params] += (flux_penalty * h_scale) * CtC_kernel

        # Solve with regularization; H is symmetric positive-definite,
        # so try Cholesky first
        H[np.diag_indices(n_total)] += ridge * h_scale
        try:
            theta = scipy_linalg.solve(H, g, assume_a='pos')
        except np.linalg.LinAlgError:
            log("SFFT: WARNING - singular normal matrix, increasing ridge")
            H[np.diag_indices(n_total)] += 1e-3 * h_scale
            theta = np.linalg.solve(H, g)

        # Reconstruct model and compute residuals
        model = _reconstruct_model(
            theta, ref, kernel_shape, kernel_poly_order, bg_poly_order, x_norm, y_norm
        )
        residual = sci - model

        # Compute RMS on good pixels
        good_mask = weight > 0
        if np.any(good_mask):
            final_rms = np.sqrt(np.mean(residual[good_mask] ** 2))
        else:
            final_rms = np.nan

        log(
            "SFFT: iteration %d/%d: n_good=%d, rms=%.4f"
            % (iteration + 1, total_iters, n_good, final_rms)
        )

        # Reweighting pass: rebuild the weights including template noise
        # propagated through the first-pass kernel, then re-solve
        if reweight_pending:
            reweight_pending = False
            kc = theta[:n_kernel_params].reshape(n_kernel_pixels, n_kpoly)
            var_conv = _propagate_template_variance(
                kc, kernel_shape, kernel_poly_order, template_err**2
            )
            if var_sci is not None:
                var_total = var_sci + var_conv
            else:
                # Science noise unknown: estimate a constant floor from the
                # residual scatter in excess of the template contribution
                abs_resid = np.abs(residual)
                robust_var = (np.median(abs_resid[good_mask]) * 1.4826) ** 2
                var_sci_const = max(robust_var - np.median(var_conv[good_mask]), 0.0)
                var_total = var_sci_const + var_conv
                log(
                    "SFFT: science noise not given, using constant variance %.4g "
                    "estimated from residuals" % var_sci_const
                )
            base_weight = _make_weight(var_total)
            # Weights changed globally — the accumulated data term is invalid
            H_data = None
            g_data = None
            log("SFFT: reweighting with kernel-propagated template noise")
            continue

        # Sigma-clipping
        if not do_clip or iteration == total_iters - 1:
            break

        if np.any(good_mask) and np.isfinite(final_rms) and final_rms > 0:
            # Use MAD for robust scale estimate
            abs_resid = np.abs(residual)
            robust_sigma = np.median(abs_resid[good_mask]) * 1.4826
            if robust_sigma > 0:
                newly_clipped = (abs_resid > sigma_clip * robust_sigma) & good_mask
                n_clipped = int(np.sum(newly_clipped))
                if n_clipped == 0:
                    log("SFFT: no pixels clipped, converged")
                    break

                good[newly_clipped] = False
                if _USE_INCREMENTAL_UPDATES:
                    # Remove just the clipped pixels' contributions instead
                    # of reassembling from the full image (exact: all terms
                    # are linear in the pixel weight)
                    ys, xs = np.nonzero(newly_clipped)
                    _downdate_normal_equations(
                        H_data, g_data, ref, sci, weight[ys, xs], ys, xs,
                        kernel_shape, kernel_poly_order, bg_poly_order, x_norm, y_norm,
                    )
                else:
                    H_data = None
                    g_data = None
                log("SFFT: clipped %d pixels (%.3f%%)" % (n_clipped, 100.0 * n_clipped / n_good))
            else:
                break
        else:
            # No usable residual statistics — further iterations would
            # re-solve the identical system
            break

    # --- Extract results ---
    kernel_coeffs = theta[:n_kernel_params].reshape(n_kernel_pixels, n_kpoly)
    bg_coeffs = theta[n_kernel_params:]
    flux_coeffs = _extract_flux_poly(theta, kernel_shape, kernel_poly_order, flux_poly_order)

    diff = sci - model
    n_good_final = int(np.sum(np.where(good, base_weight, 0.0) > 0))

    log(
        "SFFT: done. Final RMS=%.4f, n_good=%d (%.1f%%)"
        % (final_rms, n_good_final, 100.0 * n_good_final / (ny * nx))
    )
    log(
        "SFFT: flux scale polynomial coeffs: %s"
        % np.array2string(flux_coeffs, precision=4, separator=', ')
    )

    return SFFTResult(
        diff=diff,
        model=model,
        kernel_coeffs=kernel_coeffs,
        bg_coeffs=bg_coeffs,
        kernel_shape=kernel_shape,
        kernel_poly_order=kernel_poly_order,
        bg_poly_order=bg_poly_order,
        flux_poly_order=flux_poly_order,
        flux_poly_coeffs=flux_coeffs,
        n_iter=n_iter_done,
        rms=float(final_rms),
        n_good=n_good_final,
        dmask=dmask,
    )


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------


def evaluate_kernel_at(result, x, y, image_shape):
    """Evaluate the spatially varying kernel at a single image position.

    The returned stamp is the convolution kernel in the usual sense:
    ``model(p) = Σ k[dy, dx] · template(p - (dy, dx))``, with the element
    ``k[hy + dy, hx + dx]`` at array index ``(hy + dy, hx + dx)`` for
    half-sizes ``hy, hx``.

    Parameters
    ----------
    result : SFFTResult
        :class:`SFFTResult` from :func:`solve`.
    x : float
        X pixel coordinate.
    y : float
        Y pixel coordinate.
    image_shape : tuple of int
        (ny, nx) of the original image.

    Returns
    -------
    numpy.ndarray
        2-D array of shape ``result.kernel_shape``.
    """
    xn, yn = _norm_xy(x, y, image_shape)

    poly_k = _poly_terms_2d(xn, yn, result.kernel_poly_order)  # (n_kpoly,)

    kernel_vals = result.kernel_coeffs @ poly_k  # (n_kernel_pixels,)
    return kernel_vals.reshape(result.kernel_shape)


def evaluate_flux_scale(result, x, y, image_shape):
    """Evaluate the flux-scale polynomial at image position(s).

    Parameters
    ----------
    result : SFFTResult
        :class:`SFFTResult` from :func:`solve`.
    x : float or array-like
        X coordinate(s), scalar or array.
    y : float or array-like
        Y coordinate(s), scalar or array.
    image_shape : tuple of int
        (ny, nx) of the original image.

    Returns
    -------
    float or numpy.ndarray
        Flux scale value(s), same shape as x/y.
    """
    xn, yn = _norm_xy(x, y, image_shape)

    poly = _poly_terms_2d(xn, yn, result.flux_poly_order)
    return np.tensordot(result.flux_poly_coeffs, poly, axes=(0, 0))


def evaluate_background(result, x, y, image_shape):
    """Evaluate the differential background model at image position(s).

    Parameters
    ----------
    result : SFFTResult
        :class:`SFFTResult` from :func:`solve`.
    x : float or array-like
        X coordinate(s), scalar or array.
    y : float or array-like
        Y coordinate(s), scalar or array.
    image_shape : tuple of int
        (ny, nx) of the original image.

    Returns
    -------
    float or numpy.ndarray
        Background value(s), same shape as x/y.
    """
    xn, yn = _norm_xy(x, y, image_shape)

    poly = _poly_terms_2d(xn, yn, result.bg_poly_order)
    return np.tensordot(result.bg_coeffs, poly, axes=(0, 0))
