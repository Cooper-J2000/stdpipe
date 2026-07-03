"""PSF-based per-source quality metrics borrowed from the ``crowdsource``
photometry pipeline (Schlafly et al., MIT-licensed).

The metrics operate on stacks of postage stamps:

``impsf`` — neighbour-subtracted data stamps (data minus the model of every
*other* source).  For an isolated source this is just the data stamp.

``im`` — raw data stamps (no neighbour subtraction).

``psf`` — model PSF stamps for each source, normalized to integrate to one.

``weight`` — inverse-sigma stamps; pixels with ``weight == 0`` are treated as
masked.

All inputs are ``(N, S, S)`` arrays.  Outputs are length-``N`` 1D arrays.

The functions are deliberately decoupled from any particular photometry
backend so they can be reused from :func:`stdpipe.photometry_psf.measure_objects_psf`,
:func:`stdpipe.photometry_measure.measure_objects_sep`, or external code.
"""

import numpy as np


# Mixture-of-Gaussians representation of the exponential galaxy profile from
# Hogg & Lang's ``the-tractor`` (also used by ``crowdsource.galconv``).  Six
# isotropic Gaussians whose amplitudes / variances reproduce a 1-pixel-scale
# exponential.
_EXP_GAL_AMP = np.array(
    [1.99485977e-04, 2.61612679e-03, 1.89726655e-02,
     1.00186544e-01, 3.68534484e-01, 5.09490694e-01]
)
_EXP_GAL_VAR = np.array(
    [1.20078965e-03, 8.84526493e-03, 3.91463084e-02,
     1.39976817e-01, 4.60962500e-01, 1.50159566e+00]
)


def neff_fwhm(stamp):
    """FWHM-like quantity of a 2D PSF stamp, equivalent to
    ``crowdsource.psf.neff_fwhm``.

    Returns ``1.18 / sqrt(pi * sum((p/norm)^2))``.  The ``1.18`` calibrates the
    n_eff-derived width to the actual FWHM of a Gaussian PSF.
    """
    stamp = np.asarray(stamp)
    norm = np.sum(stamp, axis=(-1, -2), keepdims=True)
    norm = norm + (norm == 0)
    return 1.18 * (np.pi * np.sum((stamp / norm) ** 2, axis=(-1, -2))) ** -0.5


def convolve_with_exp_galaxy(psf_stack, re):
    """Convolve each ``(S, S)`` PSF in ``psf_stack`` with a circular exponential
    galaxy of effective radius ``re`` (in pixels).

    Uses an FFT-based 6-Gaussian mixture approximation of the exponential, so
    the result is the same shape as the input.  ``re`` may be a scalar or a
    length-N array (one per source).
    """
    psf_stack = np.asarray(psf_stack, dtype='f4')
    one_d = psf_stack.ndim == 2
    if one_d:
        psf_stack = psf_stack[None, ...]
    N, H, W = psf_stack.shape
    re = np.atleast_1d(re).astype('f4')
    if re.size == 1:
        re = np.full(N, re[0], dtype='f4')

    u = np.fft.rfftfreq(W).astype('f4')
    v = np.fft.fftfreq(H).astype('f4')
    rr2 = (u[None, :] ** 2 + v[:, None] ** 2)  # (H, W//2+1)

    # F_gal[n, v, u] = sum_k amp_k exp(-2 pi^2 re_n^2 var_k (u^2+v^2))
    coeff = -2.0 * np.pi ** 2 * (re ** 2)[:, None]   # (N, K)
    coeff = coeff * _EXP_GAL_VAR[None, :].astype('f4')  # (N, K)
    # Build Fgal per source by summing the K Gaussians.
    Fgal = np.zeros((N, rr2.shape[0], rr2.shape[1]), dtype='c8')
    for k in range(_EXP_GAL_AMP.size):
        Fgal += _EXP_GAL_AMP[k] * np.exp(coeff[:, k][:, None, None] * rr2[None, :, :])

    P = np.fft.rfft2(psf_stack)
    G = np.fft.irfft2(Fgal * P, s=(H, W)).astype('f4')

    if one_d:
        G = G[0]
    return G


def _normalize_psf(psf_stack):
    """Return PSF stacks scaled so each stamp integrates to 1 (in-place safe)."""
    psf = np.asarray(psf_stack, dtype='f4')
    norm = psf.sum(axis=(-1, -2), keepdims=True)
    norm = norm + (norm == 0) * 1e-30
    return psf / norm


def compute_qf(psf_stack, weight_stack):
    """Per-source PSF quality factor.

    ``qf = sum(P_normed * (W > 0))`` — the fraction of the PSF footprint that
    overlaps unmasked pixels.  ``1.0`` for a fully unmasked source, ``< 1`` if
    pixels are masked or off-image.
    """
    P = _normalize_psf(psf_stack)
    return np.sum(P * (np.asarray(weight_stack) > 0), axis=(-1, -2)).astype('f4')


def compute_fracflux(impsf_stack, im_stack, psf_stack, weight_stack):
    """Per-source fraction-of-flux metric.

    ``fracflux = sum(impsf * (W > 0) * P_normed) / sum(im * (W > 0) * P_normed)``

    Approaches 1 for an isolated source whose neighbours have been cleanly
    subtracted, and drops below 1 when nearby sources contribute flux at the
    target's PSF footprint.  Useful as a crowding / blendiness indicator.
    """
    P = _normalize_psf(psf_stack)
    w = (np.asarray(weight_stack) > 0).astype('f4')
    num = np.sum(impsf_stack * w * P, axis=(-1, -2))
    den = np.sum(im_stack * w * P, axis=(-1, -2))
    den_safe = den + (den == 0) * 1e-30
    return (num / den_safe).astype('f4')


def compute_spread_model(impsf_stack, psf_stack, weight_stack, fwhm=None):
    """Per-source ``spread_model`` star/galaxy classifier (and its uncertainty).

    Equivalent to SExtractor's spread_model and to ``crowdsource_base.spread_model``.
    Convolves each PSF with a tiny exponential galaxy (re = fwhm / 16 * 1.6783)
    and compares the data's similarity to PSF vs PSF-convolved-galaxy.  Returns
    ``(spread, dspread)``: a star yields ``spread ~ 0``, an extended source
    ``spread > 0``.

    If ``fwhm`` is None it is estimated per-source from the PSF stamp via
    :func:`neff_fwhm`.
    """
    psf = np.asarray(psf_stack, dtype='f4')
    impsf = np.asarray(impsf_stack, dtype='f4')
    w2 = np.asarray(weight_stack, dtype='f4') ** 2

    if fwhm is None:
        fwhm = neff_fwhm(psf)
    fwhm = np.atleast_1d(np.asarray(fwhm, dtype='f4'))
    sigma = fwhm / 16.0
    re = sigma * 1.67834699  # exponential scale length matching that sigma

    G = convolve_with_exp_galaxy(psf, re)

    # Compute dot products in float64 — for bright sources the cross-terms in
    # the dspread numerator (PWp**2 * GWG, etc.) easily overflow float32.
    psf64 = psf.astype('f8')
    impsf64 = impsf.astype('f8')
    G64 = G.astype('f8')
    w2_64 = w2.astype('f8')

    GWp = np.sum(G64 * w2_64 * impsf64, axis=(-1, -2))
    PWp = np.sum(psf64 * w2_64 * impsf64, axis=(-1, -2))
    GWP = np.sum(G64 * w2_64 * psf64, axis=(-1, -2))
    PWP = np.sum(psf64 * psf64 * w2_64, axis=(-1, -2))
    GWG = np.sum(G64 * G64 * w2_64, axis=(-1, -2))

    PWp_safe = PWp + (PWp == 0) * 1e-30
    PWP_safe = PWP + (PWP == 0) * 1e-30

    spread = (GWp / PWp_safe - GWP / PWP_safe).astype('f4')
    dspread = np.sqrt(
        np.clip(PWp ** 2 * GWG + GWp ** 2 * PWP - 2 * GWp * PWp * GWP, 0.0, np.inf)
        / (PWp_safe ** 4)
    ).astype('f4')

    return spread, dspread


def compute_psf_quality(impsf_stack, im_stack, psf_stack, weight_stack, fwhm=None):
    """Convenience wrapper: returns dict with ``qf``, ``fracflux``, ``spread_model``,
    and ``dspread_model`` for each source in the input stacks.
    """
    spread, dspread = compute_spread_model(impsf_stack, psf_stack, weight_stack, fwhm=fwhm)
    return {
        'qf': compute_qf(psf_stack, weight_stack),
        'fracflux': compute_fracflux(impsf_stack, im_stack, psf_stack, weight_stack),
        'spread_model': spread,
        'dspread_model': dspread,
    }
